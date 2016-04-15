#!/usr/bin/env th

-- This script run the optim algorithm over data and model

require 'pl'
require 'torch'

-- options from command line
-- export this `opt` to other scripts
opt = lapp[[
  -r,--learningRate       (default 1e-3)      learning rate
  -d,--learningRateDecay  (default 1e-7)      learning rate decay
  -w,--weightDecay        (default 1e-4)      weight decay
  -m,--momentum           (default 0.9)       momentum
  -b,--batchSize          (default 128)       mini-batch size
  -t,--threads            (default 4)         number of threads
  -p,--type               (default float)     float or cuda
  -i,--devid              (default 1)         gpu device id if use cuda
  -s,--seed               (default 1)         seed
  -e,--maxEpoch           (default 100)       max epoches
  -n,--snapshotIter       (default 10)        snapshot iteration
  -o,--save               (default result)    result folder
]]

-- print opt
print(sys.COLORS.red .. '==> options')
print(opt)

-- set num thread
torch.setnumthreads(opt.threads)
-- set seed
torch.manualSeed(opt.seed)
-- set defualt tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- mkdir
paths.mkdir(opt.save)

-- if use cuda
if opt.type == 'cuda' then
  print(sys.COLORS.red .. '==> switching to GPU')

  -- load cunn
  require 'cunn'
  cutorch.setDevice(opt.devid)
  print(sys.COLORS.red .. '==> use GPU #' .. cutorch.getDevice())
end

-- load data
local data = require 'data'

-- load train and test function
local train = require 'train'
local test = require 'test'

-- load model
local tModel = require 'model'
local model = tModel.model

-- print model layout
print(sys.COLORS.red .. '==> model layout')
print(model)

-- load plot function
local plot = require('plot')

-- epoch
local epoch

-- loss of train and test
local trainLoss = {}
local testLoss = {}

-- run train and test
while true do
  -- epoch tracker
  epoch = epoch or 1

  print('')
  print(sys.COLORS.red .. '==> epoch #' .. epoch .. ', batchSize = ' .. opt.batchSize)

  -- train on trainData
  print(sys.COLORS.red .. '==> training')
  train(data.trainData)

  -- test on trainData
  print(sys.COLORS.red .. '==> testing over trainData')
  local e1 = test(data.trainData)
  trainLoss[#trainLoss + 1] = e1

  -- test on testData
  print(sys.COLORS.red .. '==> testing over testData')
  local e2 = test(data.testData)
  testLoss[#testLoss + 1] = e2

  print('==> testing result of epoch ' .. sys.COLORS.red .. '#' .. epoch)
  print('==> training loss = ' .. sys.COLORS.green .. e1)
  print('==> test     loss = ' .. sys.COLORS.green .. e2)

  -- snapshot if needed
  if epoch % opt.snapshotIter == 0 then
    print(sys.COLORS.red .. '==> snapshot')
    -- save model
    local fout = paths.concat(opt.save, 'model.t7')
    local lightModel = model:clone()
    lightModel:clearState()
    torch.save(fout, lightModel)

    -- plot loss curve
    plot(trainLoss, testLoss)
  end

  -- next
  epoch = epoch + 1

  -- check if should terminate
  if epoch > opt.maxEpoch then
    break
  end
end

-- finally snapshot again
-- save model
local fout = paths.concat(opt.save, 'model.t7')
local lightModel = model:clone()
lightModel:clearState()
torch.save(fout, lightModel)

-- plot loss curve
plot(trainLoss, testLoss)
