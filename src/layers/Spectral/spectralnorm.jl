"""
    SpectralConv(kernel_shape,
                iterations, 
                norm_multiplier, 
                [init, device])
    SpectralConv(left_singular_vector, 
                right_singular_vector, 
                iterations, 
                norm_multiplier)

Spectral normalization is a simple regularization technique that 
prevents the weights in a neural network from having large spectral norms.
High sensitivity to input data degrades model performance. Though we 
can use existing regularization techniques to make the model robust against 
perturbations in training data, robustness to perturbations in test data is 
required. 

Reference -- https://arxiv.org/abs/1705.10941

# Fields 
- `left_singular_vector`: Power iteration approximation of left singular vector 
- `right_singular_vector`: Power iteration approximation of left singular vector 
- `iterations`: Number of updates to left and right singular vectors 
- `norm_multiplier`: Multiplication constant for the spectral norm term 

# Arguments 
- `kernel_shape::NTuple{N,Integer}`: weight matrix dimensions, eg, (5, 5) 
- `iterations::Integer`: Number of iterations for power approximation 
- `norm_multiplier`: Multiplicative constant for spectral norm term 

"""
struct SpectralNormalizer{U,V}
    left_singular_vector::U
    right_singular_vector::V
    iterations::Int
    norm_multiplier::Float32
end

function SpectralNormalizer(
    kernel_shape;
    iterations::Int = 1,
    norm_multiplier::Float32 = 0.95,
    init = glorot_uniform,
    device = cpu,
)
    # Necessary shapes 
    # Expecting this to be of shape (out_dim, in_dim)
    # For Conv layers, we expect the shape to be (out_channels, :)
    # U and V vectors
    right_singular_vector = device(init((kernel_shape[1], 1)))
    left_singular_vector = device(init((prod(kernel_shape[2:end]), 1)))

    return SpectralNormalizer(
        left_singular_vector,
        right_singular_vector,
        iterations,
        norm_multiplier,
    )
end

function (snl::SpectralNormalizer)(kernel)
    # update
    v_hat = snl.left_singular_vector
    u_hat = snl.right_singular_vector
    for iteration = 1:snl.iterations
        # Update the left singular vector 
        v_hat = u_hat' * kernel # shape - (1, in_dim)
        v_hat = v_hat ./ norm(v_hat) # shape - (1, in_dim)
        # Update the right singular vector 
        u_hat = v_hat * kernel'
        u_hat = u_hat ./ norm(u_hat) # shape - (out_dim, 1) 
    end

    # Update the left, right singular vectors in the struct 
    Flux.Zygote.@ignore copyto!(snl.right_singular_vector, u_hat)
    Flux.Zygote.@ignore copyto!(snl.left_singular_vector, v_hat)

    sigma = u_hat * kernel * v_hat'
    # Bound the spectral norm 
    bounded_kernel =
        all((snl.norm_multiplier ./ sigma) .< 1) ?
        (snl.norm_multiplier ./ sigma) .* kernel : kernel
    return bounded_kernel
end
