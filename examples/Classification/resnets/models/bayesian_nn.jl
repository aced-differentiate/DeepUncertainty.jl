using Flux
using Flux: @functor
using DeepUncertainty

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

    conv_layer = VariationalConv(
        kernelsize,
        Int(inplanes) => Int(outplanes),
        activations.conv;
        device = gpu,
        kwargs...,
    )
    push!(layers, conv_layer)
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

basicblock(inplanes, outplanes, downsample = false) =
    downsample ?
    Chain(
        conv_bn((3, 3), inplanes, outplanes[1]; stride = 2, pad = 1, bias = false)...,
        conv_bn(
            (3, 3),
            outplanes[1],
            outplanes[2],
            identity;
            stride = 1,
            pad = 1,
            bias = false,
        )...,
    ) :
    Chain(
        conv_bn((3, 3), inplanes, outplanes[1]; stride = 1, pad = 1, bias = false)...,
        conv_bn(
            (3, 3),
            outplanes[1],
            outplanes[2],
            identity;
            stride = 1,
            pad = 1,
            bias = false,
        )...,
    )


bottleneck(inplanes, outplanes, downsample = false) =
    downsample ?
    Chain(
        conv_bn((1, 1), inplanes, outplanes[1]; stride = 2, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...,
    ) :
    Chain(
        conv_bn((1, 1), inplanes, outplanes[1]; stride = 1, bias = false)...,
        conv_bn((3, 3), outplanes[1], outplanes[2]; stride = 1, pad = 1, bias = false)...,
        conv_bn((1, 1), outplanes[2], outplanes[3], identity; stride = 1, bias = false)...,
    )

skip_projection(inplanes, outplanes, downsample = false) =
    downsample ?
    Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 2, bias = false)...) :
    Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 1, bias = false)...)


function skip_identity(inplanes, outplanes)
    if outplanes > inplanes
        return Chain(
            MaxPool((1, 1), stride = 2),
            y -> cat(
                y,
                zeros(eltype(y), size(y, 1), size(y, 2), outplanes - inplanes, size(y, 4));
                dims = 3,
            ),
        )
    else
        return identity
    end
end
skip_identity(inplanes, outplanes, downsample) = skip_identity(inplanes, outplanes)

function resnet(
    block,
    residuals::NTuple{2,Any},
    connection = (x, y) -> @. relu(x) + relu(y);
    channel_config,
    block_config,
    nclasses = 1000,
)
    inplanes = 64
    baseplanes = 64
    layers = []
    append!(layers, conv_bn((7, 7), 3, inplanes; stride = 2, pad = (3, 3)))
    push!(layers, MaxPool((3, 3), stride = (2, 2), pad = (1, 1)))
    for (i, nrepeats) in enumerate(block_config)
        # output planes within a block
        outplanes = baseplanes .* channel_config
        # push first skip connection on using first residual
        # downsample the residual path if this is the first repetition of a block
        push!(
            layers,
            Parallel(
                connection,
                block(inplanes, outplanes, i != 1),
                residuals[1](inplanes, outplanes[end], i != 1),
            ),
        )
        # push remaining skip connections on using second residual
        inplanes = outplanes[end]
        for _ = 2:nrepeats
            push!(
                layers,
                Parallel(
                    connection,
                    block(inplanes, outplanes, false),
                    residuals[1](inplanes, outplanes[end], false),
                ),
            )
            inplanes = outplanes[end]
        end
        # next set of output plane base is doubled
        baseplanes *= 2
    end

    return Chain(
        Chain(layers...),
        Chain(AdaptiveMeanPool((1, 1)), flatten, Dense(inplanes, nclasses)),
    )
end


resnet(block, shortcut_config::Symbol, args...; kwargs...) =
    (shortcut_config == :A) ?
    resnet(block, (skip_identity, skip_identity), args...; kwargs...) :
    (shortcut_config == :B) ?
    resnet(block, (skip_projection, skip_identity), args...; kwargs...) :
    (shortcut_config == :C) ?
    resnet(block, (skip_projection, skip_projection), args...; kwargs...) :
    error(
        "Unrecognized shortcut config == $shortcut_config passed to resnet (use :A, :B, or :C).",
    )

const resnet_config = Dict(
    :resnet18 => ([1, 1], [2, 2, 2, 2], :A),
    :resnet34 => ([1, 1], [3, 4, 6, 3], :A),
    :resnet50 => ([1, 1, 4], [3, 4, 6, 3], :B),
    :resnet101 => ([1, 1, 4], [3, 4, 23, 3], :B),
    :resnet152 => ([1, 1, 4], [3, 8, 36, 3], :B),
)


struct ResNet
    layers
end

function ResNet(channel_config, block_config, shortcut_config; block, nclasses = 1000)
    layers = resnet(
        block,
        shortcut_config;
        channel_config = channel_config,
        block_config = block_config,
        nclasses = nclasses,
    )

    ResNet(layers)
end

@functor ResNet

(m::ResNet)(x) = m.layers(x)

backbone(m::ResNet) = m.layers[1]
classifier(m::ResNet) = m.layers[2]


function VariationalResNet18(; pretrain = false, nclasses = 1000)
    model = ResNet(resnet_config[:resnet18]...; block = basicblock, nclasses = nclasses)
    return model
end

function VariationalMCResNet34(; pretrain = false, nclasses = 1000)
    model = ResNet(resnet_config[:resnet34]...; block = basicblock, nclasses = nclasses)
    return model
end
