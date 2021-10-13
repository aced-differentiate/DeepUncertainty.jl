@testset "VariationalDense" begin
    input = rand(5, 32)
    layer = VariationalDense(5, 10)
    output = layer(input)
    @test size(output) == (10, 32)

    # Test gradients 
    layer = VariationalDense(2, 5)
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

@testset "VariationalConv" begin
    # Test gradients 
    layer = VariationalConv((5, 5), 3 => 6, relu)
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
