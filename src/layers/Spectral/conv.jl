"""
    SpectralConv(filter, in => out, σ = identity;
                    stride = 1, pad = 0, dilation = 1, 
                    groups = 1, 
                    iterations = 1, 
                    norm_multiplier=0.95, 
                    [bias, weight, init])
    SpectralConv(σ, weight, bias,
                    stride, pad, dilation, groups, 
                    spectralnormalizer)

Creates a convolutional layer with spectral normalization applied 
to it's weight matrix.

Reference - https://arxiv.org/abs/1705.10941

# Fields 
- `σ`: Activation function, applies to logits after layer transformation 
- `weight`: A trainable distribution from which weights are sampled 
                    in every forward pass 
- `bias`: A trainable distribution from which biases are sampled in 
                    every forward pass 
- `stride`: Convolution stride 
- `pad`
- `dilation`
- `groups`
- `spectral_normalizer`

# Arguments 
- `filter::NTuple{N,Integer}`: Kernel dimensions, eg, (5, 5) 
- `ch::Pair{<:Integer,<:Integer}`: Input channels => output channels 
- `σ::F=identity`: Activation of the dense layer, defaults to identity
- `iterations::Integer=1`: Number of iterations for power approximation
- `norm_multiplier::Float32=0.95`: Multiplicative constant for spectral norm term 

"""
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
