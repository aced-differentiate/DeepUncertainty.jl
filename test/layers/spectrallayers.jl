@testset "SpectralDense" begin
    input = rand(5, 32)
    layer = SpectralDense(5, 10)
    output = layer(input)
    @test size(output) == (10, 32)

    # Test gradients 
    layer = SpectralDense(2, 5)
    i = rand(2, 4)
    y = ones(5, 4)
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
    end
end

@testset "SpectralConv" begin
    # Test gradients 
    layer = SpectralConv((5, 5), 3 => 6, relu)
    i = rand(32, 32, 3, 4)
    y = rand(28, 28, 6, 4)
    grads = gradient(params(layer)) do
        ŷ = layer(i)
        return Flux.logitcrossentropy(ŷ, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
    end
end
