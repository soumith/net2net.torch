require 'torch'
require 'nn'

local n2n = {}

local function batchnormWarning(net)
   local printed = false
   net:apply(function(m)
	 if torch.type(m):find('BatchNormalization') then
	    if printed == false then
	       print("WARNING: this package currently does not" 
			.. " handle batchnorm based networks")
	       printed = true
	    end
	 end
   end)
end

-- net  = network
-- pos1 = position at which one has to widen the output
-- pos2 = position at which the next weight layer is present
-- newWidth   = new width of the layer
n2n.wider = function(net, pos1, pos2, newWidth)
   batchnormWarning(net)
   local m1 = net:get(pos1)
   local m2 = net:get(pos2)

   -- TODO: handle convolution cases later
   assert(torch.type(m1) == 'nn.Linear')
   assert(torch.type(m2) == 'nn.Linear')

   local w1 = m1.weight
   local w2 = m2.weight
   local b1 = m1.bias
   local b2 = m2.bias

   assert(w2:size(2) == w1:size(1), 'failed sanity check')
   assert(newWidth > w1:size(1), 'new width should be greater than old width')

   local oldWidth = w2:size(2)
   
   local nw1 = m1.weight.new() -- new weight1
   nw1:resize(newWidth, w1:size(2))

   local nw2 = m2.weight.new() -- new weight2
   nw2:resize(w2:size(1), newWidth)

   -- copy the original weights over
   nw1:narrow(1, 1, oldWidth):copy(w1)

   -- now do random selection on new weights
   local tracking = {}
   for i = oldWidth + 1, newWidth do
      local j = torch.random(1, oldWidth)
      tracking[j] = tracking[j] or {j}
      table.insert(tracking[j], i)

      -- copy the weights
      nw1:select(1, i):copy(w1:select(1, j))

   end

   -- renormalize the weights
   for k, v in pairs(tracking) do
      for kk, vv in ipairs(v) do
	 nw1[vv]:div(#v)
      end
   end
   
   return net
end

n2n.deeper = function()
   batchnormWarning(net)
end

return n2n
