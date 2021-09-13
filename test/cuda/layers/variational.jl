@testset "Variational Dense Layer" begin
    input = gpu(rand(5, 32))
    layer = gpu(VariationalDense(5, 10))
    output = layer(input)
    @test size(output) == (10, 32)

    # Test gradients 
    layer = gpu(VariationalDense(2, 5))
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

@testset "Variational Conv Layer" begin
    # Test gradients 
    layer = gpu(VariationalConv((5, 5), 3 => 6, relu))
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
