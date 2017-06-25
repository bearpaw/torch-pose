local nnlib = require('cudnn')
local conv = nnlib.SpatialConvolution
local batchnorm = nnlib.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut,dropout,stride,type)
    local s = nn.Sequential()
    if type ~= 'no_preact' then
        s:add(batchnorm(numIn))
        s:add(relu(true))        
    end
    s:add(conv(numIn,numOut/2,1,1))
    s:add(batchnorm(numOut/2))
    s:add(relu(true))
    s:add(conv(numOut/2,numOut/2,3,3,stride,stride,1,1))
    s:add(batchnorm(numOut/2))
    s:add(relu(true))
    s:add(conv(numOut/2,numOut,1,1))
    if dropout > 0 then
        s:add(nn.Dropout(dropout,nil,true))
    end
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
function Residual(numIn,numOut,dropout,stride,type, useConv)
    local dropout = dropout or 0
    local stride = stride or 1
    local type = type or 'preact'
    local useConv = useConv or false
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut,dropout,stride,type))
            :add(skipLayer(numIn,numOut,stride,useConv)))
        :add(nn.CAddTable(true))
end

return Residual
