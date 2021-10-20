struct SpectralConv{N,M,F,A,V,S}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
    spectralnormalizer::S
end

function SpectralConv(
    k::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer},
    σ = identity;
    stride = 1,
    pad = 0,
    dilation = 1,
    groups = 1,
    init = glorot_normal,
    iterations = 1,
    norm_multiplier = 0.95,
    weight = Flux.convfilter(k, (ch[1] ÷ groups => ch[2])),
    bias = true,
    device = cpu,
) where {N}

    stride = Flux.expand(Val(N), stride)
    dilation = Flux.expand(Val(N), dilation)
    pad = Flux.calc_padding(SpectralConv, pad, size(weight)[1:N], dilation, stride)
    bias = create_bias(weight, bias, size(weight, N + 2))
    norm_multiplier = convert(Float32, norm_multiplier)

    kernel_shape = size(reshape(weight, ch[2], :))
    spectralnormalizer = SpectralNormalizer(
        kernel_shape,
        iterations = iterations,
        norm_multiplier = norm_multiplier,
        init = init,
        device = device,
    )

    return SpectralConv(σ, weight, bias, stride, pad, dilation, groups, spectralnormalizer)
end

@functor SpectralConv (weight, bias)

function (c::SpectralConv)(x)
    # update
    kernel_shape = size(c.weight)
    out_channels = kernel_shape[end]
    kernel = reshape(c.weight, out_channels, :)
    # Bound the kernel 
    bounded_kernel = c.spectralnormalizer(kernel)
    bounded_kernel = reshape(bounded_kernel, kernel_shape)

    σ, b = c.σ, reshape(c.bias, ntuple(_ -> 1, length(c.stride))..., :, 1)
    output = Flux.NNlib.conv(
        x,
        bounded_kernel;
        stride = c.stride,
        pad = c.pad,
        dilation = c.dilation,
        groups = c.groups,
    )
    σ.(output .+ b)
end
