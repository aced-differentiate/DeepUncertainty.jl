"""
    SpectralDense(in, out, σ=identity;
                    iterations=1,
                    norm_multiplier=0.95, 
                    bias=true)
    SpectralDense(weight, bias, spectral_normalizer, act)

Creates a dense layer with spectral normalization applied 
to it's weight matrix.

Reference - https://arxiv.org/abs/1705.10941

# Fields 
- `weight`
- `bias`
- `spectral_normalizer`
- `act`: Activation function

# Arguents 
- `in::Integer`: Input dimension size 
- `out::Integer`: Output dimension size 
- `σ`: Acivation function, defaults to identity 
- `init`: Distribution parameters Initialization, defaults to glorot_normal
- `iterations::Integer=1`: Number of iterations for power approximation
- `norm_multiplier::Float32=0.95`: Multiplicative constant for spectral norm term 

"""
struct SpectralDense{W,B,S,F}
    weight::W
    bias::B
    spectralnormalizer::S
    act::F
end

function SpectralDense(
    in::Integer,
    out::Integer,
    σ = identity;
    bias = true,
    iterations::Integer = 1,
    norm_multiplier = 0.95,
    init = glorot_uniform,
    device = cpu,
)
    weight = init(out, in)
    b = Flux.create_bias(weight, bias, size(weight, 1))
    norm_multiplier = convert(Float32, norm_multiplier)

    kernel_shape = size(weight)
    spectralnormalizer = SpectralNormalizer(
        kernel_shape,
        iterations = iterations,
        norm_multiplier = norm_multiplier,
        init = init,
        device = device,
    )

    return SpectralDense(weight, b, spectralnormalizer, σ)
end

@functor SpectralDense

function (sd::SpectralDense)(x; eps = 1e-12)
    # Spectral regularize the weight
    bounded_kernel = sd.spectralnormalizer(sd.weight)
    logits = bounded_kernel * x .+ sd.bias
    return sd.act.(logits)
end
