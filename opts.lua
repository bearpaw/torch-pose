--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',         '',         'Path to dataset')
   cmd:option('-dataset',      'imagenet', 'Options: mpii | mpiijson | coco')
   cmd:option('-manualSeed',   0,          'Manually set RNG seed')
   cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
   cmd:option('-backend',      'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',        'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',          'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',     2,          'number of data loading threads')
   cmd:option('-inputRes',     256,        'Input image resolution')
   cmd:option('-outputRes',    64,         'Output heatmap resolution')
   cmd:option('-scaleFactor',  .25,        'Degree of scale augmentation')
   cmd:option('-rotFactor',    30,         'Degree of rotation augmentation')
   cmd:option('-rotProbab',    .4,         'Degree of rotation augmentation')
   cmd:option('-flipFactor',   .5,         'Degree of flip augmentation')
   cmd:option('-saturation',   .5,         'Degree of saturation augmentation')
   cmd:option('-brightness',   .5,         'Degree of brightness augmentation')
   cmd:option('-contrast',     .5,         'Degree of contrast augmentation')
   cmd:option('-colorJitter',  'false',    'Input image color jittering')
   cmd:option('-colorPerturb', 'true',     'Input image color jittering')
   cmd:option('-noise',        0,          'Input image color jittering')
   cmd:option('-blur',         0,          'Input image color jittering')
   cmd:option('-minusMean',    'true',     'Minus image mean')
   cmd:option('-gsize',        1,          'Kernel size to generate the Gassian-like labelmap')
   cmd:option('-mask',         0,          'Probability to mask a body parts')
   ------------- Training options --------------------
   cmd:option('-nEpochs',      0,          'Number of total epochs to run')
   cmd:option('-epochNumber',  1,          'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',    1,          'mini-batch size (1 = pure stochastic)')
   cmd:option('-iterSize',     1,          'Accumulate gradients across [iterSize] batches in training')
   cmd:option('-testOnly',     'false',    'Run on validation set only')
   cmd:option('-testRelease',  'false',    'Run on testing set only')
   cmd:option('-tenCrop',      'false',    'Ten-crop testing')
   cmd:option('-crit',         'MSE',      'Criterion type: MSE | CrossEntropy')
   cmd:option('-optMethod',    'rmsprop',  'Optimization method: rmsprop | sgd | nag | adadelta | adam')
   cmd:option('-snapshot',     1,          'How often to take a snapshot of the model (0 = never)')
   cmd:option('-debug',        'false',    'Visualize training/testing samples')
   ------------- Checkpointing options ---------------
   cmd:option('-save',         'checkpoints','Directory in which to save checkpoints')
   cmd:option('-expID',        'default',  'Experiment ID')
   cmd:option('-resume',       'none',     'Resume from the latest checkpoint in this directory')
   cmd:option('-loadModel',    'none',     'Load model')  
   ---------- Optimization options ----------------------
   cmd:option('-LR',           2.5e-4,     'initial learning rate')
   cmd:option('-momentum',     0.0,        'momentum')
   cmd:option('-weightDecay',  0.0,        'weight decay')
   cmd:option('-alpha',        0.99,       'Alpha')
   cmd:option('-epsilon',      1e-8,       'Epsilon')
   cmd:option('-dropout',      0,          'Dropout ratio')
   cmd:option('-init',         'none',     'Weight initialization method: none | heuristic | xavier | xavier_caffe | kaiming')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'hg-stacked','Options: resnet | preresnet | hg-stacked')
   cmd:option('-depth',        34,         'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',         'Options: A | B | C')
   cmd:option('-retrain',      'none',     'Path to model to retrain with')
   cmd:option('-optimState',   'none',     'Path to an optimState to reload from')
   cmd:option('-nStack',       8,          'Number of stacks in the provided hourglass model (for hg-generic)')
   cmd:option('-nFeats',       256,        'Number of features in the hourglass (for hg-generic)')
   cmd:option('-nResidual',    1,          'Number of residual module in the hourglass (for hg-generic)')
   cmd:option('-nPyraFeatsRatio',8,        'Number of features in each level of feature pyramid (nFeats/nPyraFeatsRatio)')
   cmd:option('-nPyra',        8,          'Number of levers of the feature pyramid')
   cmd:option('-nAttention',   1,         'Number of additional GT target')
   cmd:option('-nBranch',      1,          'Number of branches')
   cmd:option('-growthRate',   24,         'growthRate of DenseBlock')
   cmd:option('-baseWidth',    6,          'ResNeXt base width', 'number')
   cmd:option('-cardinality',  30,         'ResNeXt cardinality', 'number')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput','false',   'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',       'false',    'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier','false',  'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',     16,         'Number of classes in the dataset')
   cmd:option('-bg',           'false',    'If true, we will have an additional fg/bg labelmap')
   cmd:option('-scales',        0,         'If true, we will have an additional fg/bg labelmap')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.testRelease = opt.testRelease ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'mpiijson-action' then
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   elseif string.find(opt.dataset, 'mpii') ~= nil or opt.dataset == 'lsp' or opt.dataset == 'coco' or opt.dataset == 'flic' or opt.dataset == 'cooking' or opt.dataset == 'iccv17supp'then      
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
