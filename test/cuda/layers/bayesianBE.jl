@testset "Variational Dense BatchEnsemble" begin
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

@testset "Variational Conv BatchEnsemble" begin
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
