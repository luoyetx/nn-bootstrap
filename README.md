nn-bootstrap
=============

bootstrap for torch-nn. Inspired by [train-face-detector](https://github.com/torch/demos/tree/master/train-face-detector).

### data.lua

Load trainData and testData in this script, impl `size()` for each data.

### model.lua

Define model and criterion.

### train.lua

Define `train()` function, use training algorithm to do one epoch training.

### test.lua

Define `test()` function, return average loss over data from this function.

### main.lua

Run `train()` and `test()` functions and do other stuffs if needed.
