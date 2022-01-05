@testset "SpectralDense" begin
    input = rand(5, 32) |> gpu
    layer = SpectralDense(5, 10, device = gpu) |> gpu
    output = layer(input)
    @test size(output) == (10, 32)

    # Test gradients 
    layer = SpectralDense(2, 5, device = gpu) |> gpu
    i = rand(2, 4) |> gpu
    y = ones(5, 4) |> gpu
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end

@testset "SpectralConv" begin
    # Test gradients 
    layer = SpectralConv((5, 5), 3 => 6, relu, device = gpu) |> gpu
    i = rand(32, 32, 3, 4) |> gpu
    y = rand(28, 28, 6, 4) |> gpu
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end
