# Torch implementation of [Net2Net: Accelerating Learning via Knowledge Transfer by Chen, Goodfellow, Shlens](http://arxiv.org/abs/1511.05641)

- Proof of concept with unit tests
- Handles batchnorm layers in conjunction with linear and convolutional layers

```lua
n2n = require 'net2net'

-- net  = network
-- pos1 = position at which one has to widen the output
-- pos2 = position at which the next weight layer is present
-- newWidth   = new width of the layer
-- batchnorm layer should be between pos1 and pos2
-- batchnorm layer is modified to maintain identity-preserving mapping
n2n.wider(net, pos1, pos2, newWidth)

-- pos = position at which the layer has to be deepened
-- nonlin = type of non-linearity to insert
-- bnormFlag = boolean flag to insert batchnorm layer before the non-linearity
-- inserted batchnorm layer maintains identity-preserving mapping
-- make a forward pass through the model before calling n2n.deeper so that batch mean and variance can be computed
n2n.deeper(net, pos, nonlin, bnormFlag)
```

Example usage in test.lua
