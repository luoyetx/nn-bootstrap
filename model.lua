-- This script define nn model and criterion

require 'nn'

-- Sequential Model
local model = nn.Sequential()

-- layer 1
-- 1x39x39 -> 20x36x36
model:add(nn.SpatialConvolutionMM(1, 20, 4, 4, 1, 1))
model:add(nn.ReLU())
-- 20x36x36 -> 20x18x18
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- layer 2
-- 20x18x18 -> 40x16x16
model:add(nn.SpatialConvolutionMM(20, 40, 3, 3, 1, 1))
model:add(nn.ReLU())
-- 40x16x16 -> 40x8x8
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- layer 3
-- 40x8x8 -> 60x6x6
model:add(nn.SpatialConvolutionMM(40, 60, 3, 3, 1, 1))
model:add(nn.ReLU())
-- 60x6x6 -> 60x3x3
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- layer 4
-- 60x3x3 -> 80x2x2
model:add(nn.SpatialConvolutionMM(60, 80, 2, 2, 1, 1))
model:add(nn.ReLU())

-- layer 5
-- 80x2x2 -> 120
model:add(nn.View(80*2*2))
model:add(nn.Linear(80*2*2, 120))
model:add(nn.ReLU())

-- layer 6
-- 120 -> 10
model:add(nn.Linear(120, 10))

-- MSE Criterion
local criterion = nn.MSECriterion()

-- if use cuda
if opt.type == 'cuda' then
  model:cuda()
  criterion:cuda()

  -- load cudnn and convert model
  require 'cudnn'

  -- if gpu memory not an issue
  cudnn.benchmark = true
  cudnn.fastest = true
  cudnn.convert(model, cudnn)
end

-- if use pre-trained model
if opt.preModel ~= 'nil' then
  require 'torch'

  print(sys.COLORS.red .. '==> load model from ' .. opt.preModel)
  model = torch.load(opt.preModel)
end

-- exports
return {
  model = model,
  criterion = criterion,
}
