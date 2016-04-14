-- This script test the model over given data

require 'xlua'
require 'torch'

-- load model
local tModel = require 'model'
local model = tModel.model
local criterion = tModel.criterion

-- load data for batch malloc
local tData = require 'data'
local testData = tData.testData

-- malloc batch size data (x, y)
-- you should change the shape of tensor if needed
local x = torch.Tensor(opt.batchSize, testData.x:size(2),
                       testData.x:size(3), testData.x:size(4))
local y = torch.Tensor(opt.batchSize, testData.y:size(2))
-- if use cuda
if opt.type == 'cuda' then
  x = x:cuda()
  y = y:cuda()
end

-- test function
-- this function should return the avg loss over all data
local function test(testData)

  -- tic/toc
  sys.tic()

  -- meta info of data
  local testDataSize = testData:size()
  local batchSize = opt.batchSize

  -- shuffle data index
  local shuffle = torch.randperm(testDataSize)

  -- real batches
  local nBatch = 0

  -- all loss
  local loss = 0

  for t = 1, testDataSize, batchSize do
    -- progress
    xlua.progress(t, testDataSize)

    -- if not enough
    if t + batchSize - 1 > testDataSize then
      break
    end

    nBatch = nBatch + 1

    -- copy data
    local idx = 1
    for i = t, t + batchSize - 1 do
      x[idx] = testData.x[shuffle[i]]
      y[idx] = testData.y[shuffle[i]]
      idx = idx + 1
    end

    -- forward
    local y_p = model:forward(x)
    local e = criterion:forward(y_p, y)

    loss = loss + e
  end

  -- tic/toc
  local time = sys.toc()
  time = time / (nBatch * batchSize)

  -- print out test time in ms
  print('\n==> test time ~= ' .. sys.COLORS.green .. (time*1000) .. ' ms/per')

  -- avg loss
  loss = loss / nBatch

  -- return avg loss
  return loss
end

-- exports
return test
