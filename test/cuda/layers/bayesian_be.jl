@testset "VariationalDenseBE" begin
    ensemble_size = 4
    samples_per_model = 4
    input_dim = 5
    output_dim = 5
    rank = 1
    inputs = rand(Float32, input_dim, samples_per_model) |> gpu
    layer =
        VariationalDenseBE(input_dim, output_dim, rank, ensemble_size; device = gpu) |> gpu
    batch_inputs = repeat(inputs, 1, ensemble_size) |> gpu
    batch_outputs = layer(batch_inputs) |> gpu

    # Test gradients 
    grads = gradient(params(layer)) do
        ŷ = layer(batch_inputs)
        return Flux.logitcrossentropy(ŷ, batch_outputs)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end

@testset "VariationalConvBE" begin
    ensemble_size = 4
    samples_per_model = 4
    input_dim = 5
    output_dim = 10
    rank = 1
    inputs = rand(Float32, 10, 10, input_dim, samples_per_model)
    layer =
        VariationalConvBE((5, 5), 5 => 10, rank, ensemble_size, relu; device = gpu) |> gpu
    batch_inputs = repeat(inputs, 1, 1, 1, ensemble_size) |> gpu
    batch_outputs = layer(batch_inputs)

    # Test gradients 
    grads = gradient(params(layer)) do
        ŷ = layer(batch_inputs)
        return Flux.logitcrossentropy(ŷ, batch_outputs)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end
