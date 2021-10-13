@testset "DenseBE GPU" begin
    ensemble_size = 4
    samples_per_model = 4
    input_dim = 5
    output_dim = 5
    rank = 1
    inputs = rand(Float32, input_dim, samples_per_model)
    layer = DenseBE(
        input_dim,
        output_dim,
        rank,
        ensemble_size;
        alpha_init = ones,
        gamma_init = ones,
    )
    layer = layer |> gpu
    batch_inputs = gpu(repeat(inputs, 1, ensemble_size))
    batch_outputs = layer(batch_inputs)
    # Do the computation in for loop to compare outputs 
    layer = layer |> cpu
    loop_outputs = []
    for i = 1:ensemble_size
        perturbed_inputs = inputs .* layer.alpha[i]
        outputs = layer.layer(perturbed_inputs) .* layer.gamma[i]
        outputs = layer.ensemble_act.(outputs .+ layer.ensemble_bias[i])
        push!(loop_outputs, outputs)
    end
    loop_outputs = Flux.batch(loop_outputs)
    loop_outputs = reshape(loop_outputs, (output_dim, samples_per_model * ensemble_size))
    @test batch_outputs isa CuArray
    @test size(batch_outputs) == size(loop_outputs)
    @test isapprox(cpu(batch_outputs), loop_outputs, atol = 0.05)

    # Test gradients 
    layer = gpu(DenseBE(2, 5, 1, 2))
    i = gpu(rand(2, 4))
    y = gpu(ones(5, 4))
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end

@testset "ConvBE GPU" begin
    ensemble_size = 4
    samples_per_model = 4
    input_dim = 5
    output_dim = 10
    rank = 1
    inputs = rand(Float32, 10, 10, input_dim, samples_per_model)
    beconv = ConvBE(
        (5, 5),
        5 => 10,
        rank,
        ensemble_size,
        relu;
        alpha_init = ones,
        gamma_init = ones,
    )
    beconv = beconv |> gpu
    batch_inputs = gpu(repeat(inputs, 1, 1, 1, ensemble_size))
    batch_outputs = beconv(batch_inputs)
    # Do the computation in for loop to compare outputs 
    beconv = beconv |> cpu
    loop_outputs = []
    for i = 1:ensemble_size
        perturbed_inputs = inputs .* beconv.alpha[i]
        outputs = beconv.layer(perturbed_inputs) .* beconv.gamma[i]
        outputs = beconv.ensemble_act.(outputs .+ beconv.ensemble_bias[i])
        push!(loop_outputs, outputs)
    end
    loop_outputs = Flux.batch(loop_outputs)
    loop_outputs_size = size(batch_outputs)
    loop_outputs = reshape(
        loop_outputs,
        (
            loop_outputs_size[1],
            loop_outputs_size[2],
            output_dim,
            samples_per_model * ensemble_size,
        ),
    )
    @test batch_outputs isa CuArray
    @test size(batch_outputs) == size(loop_outputs)
    @test isapprox(cpu(batch_outputs), loop_outputs, atol = 0.05)

    # Test gradients 
    layer = gpu(ConvBE((5, 5), 3 => 6, 1, 4, relu))
    i = gpu(rand(32, 32, 3, 4))
    y = gpu(rand(28, 28, 6, 4))
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end
