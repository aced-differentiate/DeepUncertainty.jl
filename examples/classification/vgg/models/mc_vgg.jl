using Flux: @functor
using DeepUncertainty

dropout = 0.2

function conv_bn(
    kernelsize,
    inplanes,
    outplanes,
    activation = relu;
    rev = false,
    initβ = Flux.zeros32,
    initγ = Flux.ones32,
    ϵ = 1f-5,
    momentum = 1f-1,
    kwargs...,
)
    layers = []

    if rev
        activations = (conv = activation, bn = identity)
        bnplanes = inplanes
    else
        activations = (conv = identity, bn = activation)
        bnplanes = outplanes
    end

    push!(
        layers,
        MCConv(
            kernelsize,
            Int(inplanes) => Int(outplanes),
            dropout,
            activations.conv;
            kwargs...,
        ),
    )
    push!(
        layers,
        BatchNorm(
            Int(bnplanes),
            activations.bn;
            initβ = initβ,
            initγ = initγ,
            ϵ = ϵ,
            momentum = momentum,
        ),
    )

    return rev ? reverse(layers) : layers
end

"""
    vgg_block(ifilters, ofilters, depth, batchnorm)

A VGG block of convolution layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `ifilters`: number of input feature maps
- `ofilters`: number of output feature maps
- `depth`: number of convolution/convolution + batch norm layers
- `batchnorm`: set to `true` to include batch normalization after each convolution
"""
function vgg_block(ifilters, ofilters, depth, batchnorm)
    k = (3, 3)
    p = (1, 1)
    layers = []
    for _ = 1:depth
        if batchnorm
            append!(layers, conv_bn(k, ifilters, ofilters; pad = p, bias = false))
        else
            push!(layers, Conv(k, ifilters => ofilters, dropout, relu, pad = p))
        end
        ifilters = ofilters
    end
    return layers
end

"""
    vgg_convolutional_layers(config, batchnorm, inchannels)

Create VGG convolution layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `config`: vector of tuples `(output_channels, num_convolutions)`
            for each block (see [`Metalhead.vgg_block`](#))
- `batchnorm`: set to `true` to include batch normalization after each convolution
- `inchannels`: number of input channels
"""
function vgg_convolutional_layers(config, batchnorm, inchannels)
    layers = []
    ifilters = inchannels
    for c in config
        append!(layers, vgg_block(ifilters, c..., batchnorm))
        push!(layers, MaxPool((2, 2)))
        ifilters, _ = c
    end
    return layers
end

"""
    vgg_classifier_layers(imsize, nclasses, fcsize, dropout)

Create VGG classifier (fully connected) layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `imsize`: tuple `(width, height, channels)` indicating the size after
            the convolution layers (see [`Metalhead.vgg_convolutional_layers`](#))
- `nclasses`: number of output classes
- `fcsize`: input and output size of the intermediate fully connected layer
- `dropout`: the dropout level between each fully connected layer
"""
function vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
    layers = []
    push!(layers, flatten)
    push!(layers, Dense(Int(prod(imsize)), fcsize, relu))
    push!(layers, Dropout(dropout))
    push!(layers, Dense(fcsize, fcsize, relu))
    push!(layers, Dropout(dropout))
    push!(layers, Dense(fcsize, nclasses))

    return layers
end

"""
    vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Create a VGG model
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments
- `imsize`: input image width and height as a tuple
- `config`: the configuration for the convolution layers
            (see [`Metalhead.vgg_convolutional_layers`](#))
- `inchannels`: number of input channels
- `batchnorm`: set to `true` to use batch normalization after each convolution
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
"""
function vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)
    conv = vgg_convolutional_layers(config, batchnorm, inchannels)
    imsize = Flux.outputsize(conv, (imsize..., inchannels); padbatch = true)[1:3]
    class = vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
    return Chain(Chain(conv...), Chain(class...))
end

const vgg_config = Dict(
    :A => [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    :B => [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    :D => [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    :E => [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)],
)

"""
    VGG(imsize = (224, 224); config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Create a `VGG` model
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`vgg`](#).

# Arguments
- `imsize`: input image width and height as a tuple
- `config`: the configuration for the convolution layers
            (see [`Metalhead.vgg_convolutional_layers`](#))
- `inchannels`: number of input channels
- `batchnorm`: set to `true` to use batch normalization after each convolution
- `nclasses`: number of output classes
- `fcsize`: intermediate fully connected layer size
            (see [`Metalhead.vgg_classifier_layers`](#))
- `dropout`: dropout level between fully connected layers
"""
struct VGG
    layers
end

function VGG(
    imsize::NTuple{2,<:Integer} = (224, 224);
    config,
    inchannels,
    batchnorm = false,
    nclasses,
    fcsize,
    dropout,
)
    layers = vgg(
        imsize;
        config = config,
        inchannels = inchannels,
        batchnorm = batchnorm,
        nclasses = nclasses,
        fcsize = fcsize,
        dropout = dropout,
    )

    VGG(layers)
end

@functor VGG

(m::VGG)(x) = m.layers(x)

backbone(m::VGG) = m.layers[1]
classifier(m::VGG) = m.layers[2]

"""
    VGG11(; pretrain = false, batchnorm = false)

Create a VGG-11 style model
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`VGG`](#).

!!! warning
    `VGG11` does not currently support pretrained weights.

# Arguments
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function VGG11(; img_size = 32, inchannels = 3, nclasses = 10)
    model = VGG(
        (img_size, img_size);
        config = vgg_config[:A],
        inchannels = inchannels,
        batchnorm = true,
        nclasses = nclasses,
        fcsize = 4096,
        dropout = 0.5,
    )
    return model
end

"""
    VGG13(; pretrain = false, batchnorm = false)

Create a VGG-11 style model
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`VGG`](#).

!!! warning
    `VGG13` does not currently support pretrained weights.

# Arguments
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function VGG13(; img_size = 32, inchannels = 3, nclasses = 10)
    model = VGG(
        (img_size, img_size);
        config = vgg_config[:B],
        inchannels = inchannels,
        batchnorm = true,
        nclasses = nclasses,
        fcsize = 4096,
        dropout = 0.5,
    )
    return model
end

"""
    VGG16(; pretrain = false, batchnorm = false)

Create a VGG-11 style model
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`VGG`](#).

!!! warning
    `VGG16` does not currently support pretrained weights.

# Arguments
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function VGG16(; img_size = 32, inchannels = 3, nclasses = 10)
    model = VGG(
        (img_size, img_size);
        config = vgg_config[:D],
        inchannels = inchannels,
        batchnorm = true,
        nclasses = nclasses,
        fcsize = 4096,
        dropout = 0.5,
    )
    return model
end

"""
    VGG19(; pretrain = false, batchnorm = false)

Create a VGG-11 style model
([reference](https://arxiv.org/abs/1409.1556v6)).
See also [`VGG`](#).

!!! warning
    `VGG19` does not currently support pretrained weights.

# Arguments
- `pretrain`: set to `true` to load pre-trained model weights for ImageNet
"""
function MCVGG19(; img_size = 32, inchannels = 3, nclasses = 10)
    model = VGG(
        (img_size, img_size);
        config = vgg_config[:E],
        inchannels = inchannels,
        batchnorm = true,
        nclasses = nclasses,
        fcsize = 4096,
        dropout = 0.5,
    )
    return model
end
