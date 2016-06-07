require 'nn'
n2n = require 'net2net'

torch.setdefaulttensortype('torch.FloatTensor')
local eps = 1e-6

----------------------------------------------------------
-- test Linear (wider)
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

-- insert BatchNormalization layer after linear layer
local m = nn.Sequential()
m:add(nn.Linear(100,200))
m:add(nn.BatchNormalization(200))
m:add(nn.ReLU())
-- adding another BatchNormalization layer here is wrong
-- uncommenting the following line will cause an error
-- m:add(nn.BatchNormalization(200))
m:add(nn.Linear(200,400))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 1000 units
n2n.wider(m, 1, 4, 1000)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

----------------------------------------------------------
-- test nn.SpatialConvolution (wider)
local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(4, 16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 3, 128)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

-- insert BatchNormalization layer after convolution
local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.SpatialBatchNormalization(32))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 4, 128)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

----------------------------------------------------------
-- test nn.SpatialConvolutionMM (wider)
local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(4, 16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 3, 128)

local outWider = m:forward(inp):clone()

assert(out:add(-1, outWider):abs():max() < eps)

-- insert BatchNormalization layer after convolution
local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.SpatialBatchNormalization(32))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- make the 2nd layer of m to 128 feature maps
n2n.wider(m, 1, 4, 128)

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

-- add a new linear layer with ReLU
n2n.deeper(m, 1, nn.ReLU(), false)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

-- re-create model
local m = nn.Sequential()
m:add(nn.Linear(100,200))
m:add(nn.ReLU())
m:add(nn.Linear(200,400))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- add a new linear layer with ReLU and BatchNormalization
n2n.deeper(m, 1, nn.ReLU(), true)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

--------------------------------------------------------
-- test Convolution (deeper)
local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(4, 16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()

-- add a new convolutional layer with ReLU
n2n.deeper(m, 1, nn.ReLU(), false)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

-- re-create model
local m = nn.Sequential()
m:add(nn.SpatialConvolution(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolution(32, 64, 3, 3))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- add a new convolutional layer with ReLU and BatchNormalization
n2n.deeper(m, 1, nn.ReLU(), true)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

--------------------------------------------------------
-- test ConvolutionMM (deeper)
local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

local inp = torch.randn(4, 16, 10, 10)

-- output before transform
local out = m:forward(inp):clone()

-- add a new convolutional layer with ReLU
n2n.deeper(m, 1, nn.ReLU(), false)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

-- re-create model
local m = nn.Sequential()
m:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
m:add(nn.ReLU())
m:add(nn.SpatialConvolutionMM(32, 64, 3, 3))
m:add(nn.ReLU())

-- output before transform
local out = m:forward(inp):clone()

-- add a new convolutional layer with ReLU and BatchNormalization
n2n.deeper(m, 1, nn.ReLU(), true)

local outDeeper = m:forward(inp):clone()

assert(out:add(-1, outDeeper):abs():max() < eps)

--------------------------------------------------------
print('Tests passed')
