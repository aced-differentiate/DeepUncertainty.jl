@testset "VariationalDense" begin
    input = gpu(rand(5, 32))
    layer = VariationalDense(5, 10, device = gpu)
    output = layer(input)
    @test size(output) == (10, 32)

    # Test gradients 
    layer = VariationalDense(2, 5, device = gpu)
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

@testset "VariationalConv" begin
    # Test gradients 
    layer = VariationalConv((5, 5), 3 => 6, relu, device = gpu)
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
