-- This script use training algorithm over trainData

require 'xlua'
require 'optim'
require 'torch'

-- load model
local tModel = require 'model'
local model = tModel.model
local criterion = tModel.criterion

-- load data for batch malloc
local tData = require 'data'
local trainData = tData.trainData

-- malloc batch size data (x, y)
-- you should change the shape of tensor if needed
local x = torch.Tensor(opt.batchSize, trainData.x:size(2),
                       trainData.x:size(3), trainData.x:size(4))
local y = torch.Tensor(opt.batchSize, trainData.y:size(2))
-- if use cuda
if opt.type == 'cuda' then
  x = x:cuda()
  y = y:cuda()
end

-- optim state
local optimState = {
  learningRate = opt.learningRate,
  learningRateDecay = opt.learningRateDecay,
  momentum = opt.momentum,
  weightDecay = opt.weightDecay,
}

-- parameters and gradParameters
local w, dE_dw = model:getParameters()

-- epoch
local epoch

-- train function
local function train(trainData)

  -- epoch tracker
  epoch = epoch or 1

  -- tic/toc
  sys.tic()

  -- meta info of data
  local trainDataSize = trainData:size()
  local batchSize = opt.batchSize

  -- shuffle data index
  local shuffle = torch.randperm(trainDataSize)

  -- real batches
  local nBatch = 0

  for t = 1, trainDataSize, batchSize do
    -- progress
    xlua.progress(t, trainDataSize)

    -- if not enough
    if t + batchSize - 1 > trainDataSize then
      break
    end

    nBatch = nBatch + 1

    -- copy data
    local idx = 1
    for i = t, t + batchSize - 1 do
      x[idx] = trainData.x[shuffle[i]]
      y[idx] = trainData.y[shuffle[i]]
      idx = idx + 1
    end

    -- eval function, return E, dE_dw
    local eval = function(w)

      -- reset gradiant
      dE_dw:zero()

      -- forward batch and get loss
      local y_p = model:forward(x)
      local E = criterion:forward(y_p, y)

      -- backward
      local dE_dy = criterion:backward(y_p, y)
      model:backward(x, dE_dy)

      -- return E, dE_dw
      return E, dE_dw
    end

    -- optim with sgd
    optim.sgd(eval, w, optimState)
  end

  -- tic/toc
  local time = sys.toc()

  -- training time of every sample
  time = time / (nBatch * batchSize)

  -- print out time in ms
  print('\n==> training time ~= ' .. sys.COLORS.green .. (time*1000) .. ' ms/per')

  -- done
end

-- exports
return train
