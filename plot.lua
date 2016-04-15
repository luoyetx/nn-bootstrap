-- This script save loss csv and plot the curve by given losses

require 'gnuplot'

-- save and plot
local function plot(trainLoss, testLoss)
  -- save loss to loss.csv
  local fout = io.open(paths.concat(opt.save, 'loss.csv'), 'w')
  -- header
  fout:write('epoch,training,test\n')
  -- content
  for i = 1, #trainLoss do
    fout:write(i .. ',' .. trainLoss[i] .. ',' .. testLoss[i] .. '\n')
  end
  fout:close()

  -- plot loss over epoch
  local epoches = torch.range(1, #trainLoss)
  trainLoss = torch.Tensor(trainLoss)
  testLoss = torch.Tensor(testLoss)

  gnuplot.pngfigure(paths.concat(opt.save, 'loss.png'))
  gnuplot.plot({'training loss', epoches, trainLoss, '-'},
               {'test loss', epoches, testLoss, '-'})
  gnuplot.xlabel('epoches')
  gnuplot.ylabel('loss')
  gnuplot.title('loss over epoches')
  gnuplot.plotflush()

  -- close
  gnuplot.closeall()
end

-- exports
return plot
