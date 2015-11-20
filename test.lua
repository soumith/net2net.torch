require 'nn'
n2n = require 'net2net'

torch.setdefaulttensortype('torch.FloatTensor')
local eps = 1e-6

----------------------------------------------------------
-- test Linear
local m = nn.Sequential()
m:add(nn.Linear(100,200))
m:add(nn.ReLU())
m:add(nn.Linear(200,400))
m:add(nn.ReLU())

local inp = torch.randn(4, 100)

-- output before transform
local out = m:forward(inp):clone()


-- make the 2nd layer of m to 1000 units
n2n.wider(m, 1, 3, 1000)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)


----------------------------------------------------------
-- test nn.SpatialConvolution
local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()


-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 3, 128)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

----------------------------------------------------------
-- test nn.SpatialConvolutionMM
local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()


-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 3, 128)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

----------------------------------------------------------
-- test Linear (deeper)
local m = nn.Sequential()
m:add(nn.Linear(100,200))
m:add(nn.ReLU())
m:add(nn.Linear(200,400))
m:add(nn.ReLU())

local inp = torch.randn(4, 100)

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 1000 units
n2n.deeper(m, 1, nn.ReLU())

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)
--------------------------------------------------------
-- test Convolution (deeper)

local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()


-- make the 2nd layer of m to 128 feature maps
n2n.deeper(m, 1, nn.ReLU())

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

--------------------------------------------------------
-- test ConvolutionMM (deeper)

local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()


-- make the 2nd layer of m to 128 feature maps
n2n.deeper(m, 1, nn.ReLU())

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)



print('Tests passed')
