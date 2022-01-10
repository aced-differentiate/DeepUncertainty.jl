using DeepUncertainty




# MC Dropout block 
function mc_conv_bn(
    kernelsize,
    inplanes,
    outplanes,
    dropout_rate = 0.2,
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

    dropout = (x; k...) -> Flux.dropout(x, dropout_rate; k...)
    conv_layer =
        Conv(kernelsize, Int(inplanes) => Int(outplanes), activations.conv; kwargs...)
    mclayer = MCLayer(conv_layer, dropout)
    push!(layers, mclayer)
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


cat_channels(x, y) = cat(x, y; dims = 3)
