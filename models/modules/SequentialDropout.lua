require 'nn'
local SequentialDropout, Parent = torch.class('nn.SequentialDropout', 'nn.Sequential')

function SequentialDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0
   self.train = true
   self.dropped = false
end

function SequentialDropout:setp(p)
   self.p = p
end

function SequentialDropout:updateOutput(input)
   local hasEqualSize = false
   if self.inputSize and #self.inputSize == input:dim() then
      hasEqualSize = true
      for i = 1,input:dim() do
         if self.inputSize[i] ~= input:size(i) then
            hasEqualSize = false
            break
         end
      end
   end
   if not hasEqualSize then
      -- record input and output size
      self.inputSize = #input
      Parent.updateOutput(self, input)
      self.outputSize = #self.output
   end

   if self.train and torch.rand(1)[1] < self.p then
      self.dropped = true
      self.output:resize(self.outputSize):zero()
   else
      self.dropped = false
      if hasEqualSize then
         Parent.updateOutput(self, input)
      end
      if not self.train then
         self.output:mul(1 - self.p)
      end
   end
   return self.output
end

function SequentialDropout:updateGradInput(input, gradOutput)
   if self.dropped then
      self.gradInput:resize(self.inputSize):zero()
   elseif not self.train then
      error('backprop only defined while training')
   else
      Parent.updateGradInput(self, input, gradOutput)
   end
   return self.gradInput
end

function SequentialDropout:accGradParameters(input, gradOutput, scale)
   if not self.train then
      error('backprop only defined while training')
   elseif not self.dropped then
      Parent.accGradParameters(self, input, gradOutput, scale)
   end
end

function SequentialDropout:backward(input, gradOutput, scale)
   if self.dropped then
      self.gradInput:resize(self.inputSize):zero()
   elseif not self.train then
      error('backprop only defined while training')
   else
      Parent.backward(self, input, gradOutput, scale)
   end
   return self.gradInput
end

function SequentialDropout:accUpdateGradParameters(input, gradOutput, lr)
   if not self.train then
      error('backprop only defined while training')
   elseif not self.dropped then
      Parent.accUpdateGradParameters(self, input, gradOutput, lr)
   end
end


function SequentialDropout:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = ('nn.SequentialDropout (%.2f)'):format(self.p)
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end