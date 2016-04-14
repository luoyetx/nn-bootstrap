-- This script load the h5 data file
-- All data should be processed out of this script

require 'hdf5'

-- train data
local trainH5 = opt.trainData or 'train.h5'
local trainData = hdf5.open(trainH5, 'r'):all()
local trainDataSize = trainData.x:size(1)
-- size
function trainData:size()
  return trainDataSize
end
-- index for nn.optim maybe
setmetatable(trainData,
  { __index = function(self, i)
                return {self.x[i], self.y[i]}
              end
  })

-- test data
local testH5 = opt.testData or 'test.h5'
local testData = hdf5.open(testH5, 'r'):all()
local testDataSize = testData.x:size(1)
-- size
function testData:size()
  return testDataSize
end
-- index for nn.optim maybe
setmetatable(testData,
  { __index = function(self, i)
                return {self.x[i], self.y[i]}
              end
  })

-- exports
return {
  trainData = trainData,
  testData = testData,
}
