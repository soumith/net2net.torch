require 'nn'
n2n = require 'net2net'

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

print(out:add(-1, outWider):abs():max())
