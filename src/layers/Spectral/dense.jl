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
