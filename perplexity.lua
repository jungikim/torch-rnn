require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local cmd = torch.CmdLine()

cmd:option('-checkpoint', '')
cmd:option('-split', 'val')
cmd:option('-gpu', -1)
cmd:option('-gpu_backend', 'cuda')

opt = cmd:parse(arg)

-- Set up GPU stuff
dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  print 'Running in CPU mode'
end

-- Load the checkpoint and model
checkpoint = torch.load(opt.checkpoint)
model = checkpoint.model
model:type(dtype)

require 'io'

local sen = io.stdin:read("*l")
while sen do
  local x = model:encode_string(sen):type(dtype)
  local senLen = x:size(1)
  x = x:reshape(1, senLen, -1)

  model:resetStates()
  local scores = model:forward(x):view(senLen, -1):type(dtype)
  scores = nn.LogSoftMax():forward(scores):type(dtype)
  local sum = 0.0
  for i=1,senLen-1 do
    sum = sum + scores[i][x[1][i+1]]
  end

  local avgNLL = -sum / senLen
  local perp = torch.exp(avgNLL)
  print(perp)

  sen = io.stdin:read("*l")
end
