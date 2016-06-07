require 'torch'
require 'nn'

local n2n = {}

-- net  = network
-- pos1 = position at which one has to widen the output
-- pos2 = position at which the next weight layer is present
-- newWidth   = new width of the layer
n2n.wider = function(net, pos1, pos2, newWidth)
   local m1 = net:get(pos1)
   local m2 = net:get(pos2)

   local w1 = m1.weight
   local w2 = m2.weight
   local b1 = m1.bias

   -- find batch normalization layers between pos1 and pos2
   local bnorm
   for i=pos1,pos2 do
      if torch.type(net:get(i)):find('BatchNormalization') then
         print(bnorm)
         if not bnorm then
            bnorm = net:get(i)
         else
            error("Multiple BatchNormalization layers found!")
         end
      end
   end

   if torch.type(m1):find('SpatialConvolution') or torch.type(m1) == 'nn.Linear' then

      if torch.type(m1) == 'nn.SpatialConvolutionMM' then
         w1 = w1:view(w1:size(1), m1.nInputPlane, m1.kH, m1.kW)
         w2 = w2:view(w2:size(1), m2.nInputPlane, m2.kH, m2.kW)
      end

      assert(w2:size(2) == w1:size(1), 'failed sanity check')
      assert(newWidth > w1:size(1), 'new width should be greater than old width')

      local oldWidth = w2:size(2)

      local nw1 = m1.weight.new() -- new weight1
      local nw2 = m2.weight.new() -- new weight2
      
      if w1:dim() == 4 then
         nw1:resize(newWidth, w1:size(2), w1:size(3), w1:size(4))
         nw2:resize(w2:size(1), newWidth, w2:size(3), w2:size(4))
      else
         nw1:resize(newWidth, w1:size(2))
         nw2:resize(w2:size(1), newWidth)
      end

      local nb1 = m1.bias.new() -- new bias
      nb1:resize(newWidth)

      -- new variables for batch normalization layer
      local nsave_mean, nsave_std, nrunning_mean, nrunning_var, nweight, nbias
      if bnorm then
         nsave_mean = bnorm.save_mean.new():resize(newWidth)
         nsave_std = bnorm.save_std.new():resize(newWidth)
         nrunning_mean = bnorm.running_mean.new():resize(newWidth)
         nrunning_std = bnorm.running_var.new():resize(newWidth)
         if bnorm.affine then
            nweight = bnorm.weight.new():resize(newWidth)
            nbias = bnorm.bias.new():resize(newWidth)
         end
      end

      w2 = w2:transpose(1, 2)
      nw2 = nw2:transpose(1, 2)

      -- copy the original weights over
      nw1:narrow(1, 1, oldWidth):copy(w1)
      nw2:narrow(1, 1, oldWidth):copy(w2)

      nb1:narrow(1, 1, oldWidth):copy(b1)

      -- copy original mean, var, and scaling and shifting factors over from batch normalization layer
      if bnorm then
         nsave_mean:narrow(1, 1, oldWidth):copy(bnorm.save_mean)
         nsave_std:narrow(1, 1, oldWidth):copy(bnorm.save_std)
         nrunning_mean:narrow(1, 1, oldWidth):copy(bnorm.running_mean)
         nrunning_std:narrow(1, 1, oldWidth):copy(bnorm.running_var)
         if bnorm.affine then
            nweight:narrow(1, 1, oldWidth):copy(bnorm.weight)
            nbias:narrow(1, 1, oldWidth):copy(bnorm.bias)
         end
      end

      -- now do random selection on new weights
      local tracking = {}
      for i = oldWidth + 1, newWidth do
         local j = torch.random(1, oldWidth)
         tracking[j] = tracking[j] or {j}
         table.insert(tracking[j], i)

         -- copy the weights
         nw1:select(1, i):copy(w1:select(1, j))
         nw2:select(1, i):copy(w2:select(1, j))

         nb1[i] = b1[j]

         -- copy mean, var, scaling and shifting factors from batch normalization layer
         -- 1D tensors, so copy won't work
         if bnorm then
            nsave_mean[i] = bnorm.save_mean[j]
            nsave_std[i] = bnorm.save_std[j]
            nrunning_mean[i] = bnorm.running_mean[j]
            nrunning_std[i] = bnorm.running_var[j]
            if bnorm.affine then
               nweight[i] = bnorm.weight[j]
               nbias[i] = bnorm.bias[j]
            end
         end
      end

      -- renormalize the weights
      for k, v in pairs(tracking) do
         for kk, vv in ipairs(v) do
            nw2[vv]:div(#v)
         end
      end

      w2 = w2:transpose(1, 2)
      nw2 = nw2:transpose(1, 2)

      m1.nOutputPlane = newWidth
      m2.nInputPlane = newWidth

      if torch.type(m1) == 'nn.SpatialConvolutionMM' then
         nw1 = nw1:view(nw1:size(1), m1.nInputPlane* m1.kH* m1.kW)
         nw2 = nw2:view(nw2:size(1), m2.nInputPlane* m2.kH* m2.kW)
      end

      m1.weight = nw1
      m2.weight = nw2

      m1.gradWeight = m1.weight:clone():zero()
      m2.gradWeight = m2.weight:clone():zero()

      m1.bias = nb1
      m1.gradBias = m1.bias:clone():zero()

      if bnorm then
         bnorm.save_mean = nsave_mean
         bnorm.save_std = nsave_std
         bnorm.running_mean = nrunning_mean
         bnorm.running_var = nrunning_std
         if bnorm.affine then
            bnorm.weight = nweight
            bnorm.bias = nbias
            bnorm.gradWeight = torch.Tensor(newWidth)
            bnorm.gradBias = torch.Tensor(newWidth)
         end
      end
      
   else
      error('Only nn.Linear and *.SpatialConvolution* supported')
   end
   return net
end

-- net  = network
-- pos = position at which the layer has to be deepened
-- nonlin = type of non-linearity to insert
-- bnormFlag = boolean flag for adding a batch normalization layer
n2n.deeper = function(net, pos, nonlin, bnormFlag)
   nonlin = nonlin or nn.Identity()

   local m = net:get(pos)
   local m2, bnorm

   if torch.type(m) == 'nn.Linear' then
      m2 = m.new(m.weight:size(1), m.weight:size(1)) -- a square linear
      m2.weight:copy(torch.eye(m.weight:size(1)))
      m2.bias:zero()
      
      if bnormFlag then
         bnorm = nn.BatchNormalization(m2.weight:size(1), 1e-5 , 0.01, true)
      end
   elseif torch.type(m):find('SpatialConvolution') then
      assert(m.kH % 2 == 1 and m.kW % 2 == 1, 'kernel height and width have to be odd')
      local padH = (m.kH - 1) / 2
      local padW = (m.kW - 1) / 2

      m2 = m.new(m.nOutputPlane, m.nOutputPlane, m.kH, m.kW, 1, 1, padH, padW) -- a square conv

      -- fill with identity
      m2.weight:zero()
      local cH = math.floor(m.kH / 2) + 1
      local cW = math.floor(m.kW / 2) + 1

      -- reshape weights if SpatialConvolutionMM
      local restore = false
      if m2.weight:dim() == 2 then
         m2.weight = m2.weight:view(m2.weight:size(1), m2.nInputPlane, m2.kH, m2.kW)
         restore = true
      end

      for i = 1, m.nOutputPlane do
         m2.weight:narrow(1, i, 1):narrow(2, i, 1):narrow(3, cH, 1):narrow(4, cW, 1):fill(1)
      end

      -- reshape weights if SpatialConvolutionMM
      if restore then
         m2.weight = m2.weight:view(m2.weight:size(1), m2.nInputPlane * m2.kH * m2.kW)
      end

      -- zero bias
      m2.bias:zero()

      if bnormFlag then
         bnorm = nn.SpatialBatchNormalization(m2.nOutputPlane, 1e-5 , 0.01, true)
      end
   else
      error('Module type not supported')
   end

   if bnormFlag then
      if net.modules[pos].output then
         local input = net.modules[pos].output
         bnorm:forward(input);
         bnorm.weight = torch.cinv(bnorm.save_std)
         bnorm.bias = bnorm.save_mean
      else
         error("Cannot compute batch norm statistics, make a forward pass before calling n2n.deeper function")
      end
   end

   local s = nn.Sequential()
   s:add(m)
   if bnormFlag then
      s:add(bnorm)
   end
   s:add(nonlin)
   s:add(m2)
   net.modules[pos] = s
   return net
end

return n2n
