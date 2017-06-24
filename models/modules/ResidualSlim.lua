local nnlib = require('cudnn')
local conv = nnlib.SpatialConvolution
local batchnorm = nnlib.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut,stride,type)
    local s = nn.Sequential()
    if type ~= 'no_preact' then
        s:add(batchnorm(numIn))
        s:add(relu(true))        
    end
    s:add(conv(numIn,numOut/4,1,1))
    s:add(batchnorm(numOut/4))
    s:add(relu(true))
    s:add(conv(numOut/4,numOut/4,3,3,stride,stride,1,1))
    s:add(batchnorm(numOut/4))
    s:add(relu(true))
    s:add(conv(numOut/4,numOut,1,1))
    return s
end

-- Skip layer
local function skipLayer(numIn,numOut,stride, useConv)
    if useConv then
        -- print('useConv')
        return nn.Sequential()
            :add(batchnorm(numIn))
            :add(relu(true))        
            :add(conv(numIn,numOut,1,1,stride,stride))
    end

    if numIn == numOut and stride == 1 then
        -- print('Identity')
        return nn.Identity()
    else
        -- print('Stride')
        return nn.Sequential()
            :add(batchnorm(numIn))
            :add(relu(true))        
            :add(conv(numIn,numOut,1,1,stride,stride))
    end
end

-- Residual block
function Residual(numIn,numOut,stride,type, useConv)
    local stride = stride or 1
    local type = type or 'preact'
    local useConv = useConv or false
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut,stride,type))
            :add(skipLayer(numIn,numOut,stride,useConv)))
        :add(nn.CAddTable(true))
end

return Residual
