-- Same as official pretrained model. No Residual before the last lin module
-- 2017-02-22 BN is not correctly used in original HG
-- in shortcut we also need bn
-- 2017-06-22: Modified from hg_v3_bn_preact_useconv.lua. Fix BN-Relu position bug
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local Residual = require('models.modules.Residual')
local nnlib = cudnn

local function hourglass(n, f, nModules, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,nModules,low1)
    else
        low2 = low1
        for i = 1,nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = Residual(f,f,1,'preact',true)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(numOut)(l))
end

local function preact(num, inp)    
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(num)(inp))
end

function createModel(opt)

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128,1,'no_preact')(cnv1) 

    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats, opt.nResidual, inter)

        -- Should do BN After HG
        local ll = preact(opt.nFeats, hg)

        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,opt.nClasses,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)

            -- tmpOut_ mapping should first BN
            local tmpOut_ = preact(opt.nClasses, tmpOut)
            tmpOut_ = nnlib.SpatialConvolution(opt.nClasses,opt.nFeats,1,1,1,1,0,0)(tmpOut_)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    -- MSR init
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
        print('conv')
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
        print('bn')
         v.weight:fill(1)
         v.bias:zero()
      end
    end

    -- ConvInit('cudnn.SpatialConvolution')
    -- ConvInit('nn.SpatialConvolution')
    -- BNInit('fbnn.SpatialBatchNormalization')
    -- BNInit('cudnn.SpatialBatchNormalization')
    -- BNInit('nn.SpatialBatchNormalization')

    return model:cuda()

end

return createModel