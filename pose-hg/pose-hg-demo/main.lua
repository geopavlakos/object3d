require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

dataset = arg[1]
set = arg[2]
a = loadAnnotations(dataset,set)

model = arg[3]

if model == 'pretrained' then
    m = torch.load('pose-hg-pascal3d.t7')   -- Load pre-trained model
else
    m = torch.load(model)   -- Load the specified model
    m:evaluate()
end
m:cuda()

idxs = torch.range(1,a.nsamples)

nsamples = idxs:nElement() 
-- Displays a convenient progress bar
xlua.progress(0,nsamples)
predHMs = torch.Tensor(1,102,64,64)

expDir = paths.concat('exp',dataset)
os.execute('mkdir -p ' .. expDir)

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image
    local im = image.load('data/' .. dataset .. '/images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)

    -- Get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    out = applyFn(function (x) return x:clone() end, out)
    local flippedOut = m:forward(flip(inp:view(1,3,256,256):cuda()))
    flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
    out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)
    cutorch.synchronize()

    predHMs:copy(out[#out])
    local predFile = hdf5.open(paths.concat(expDir, set .. '_' .. idxs[i] .. '.h5'), 'w')
    predFile:write('heatmaps', predHMs)
    predFile:close()

    xlua.progress(i,nsamples)

    collectgarbage()
end
