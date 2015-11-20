# Torch implementation of [Net2Net: Accelerating Learning via Knowledge Transfer by Chen, Goodfellow, Shlens](http://arxiv.org/abs/1511.05641)

- Proof of concept with unit tests
- Does not handle batchnorm cases yet

```lua
n2n = require 'net2net'

-- net  = network
-- pos1 = position at which one has to widen the output
-- pos2 = position at which the next weight layer is present
-- newWidth   = new width of the layer
n2n.wider(net, pos1, pos2, newWidth)

-- pos = position at which the layer has to be deepened
-- nonlin = type of non-linearity to insert
n2n.deeper(net, pos, nonlin)
```

Example usage in test.lua



