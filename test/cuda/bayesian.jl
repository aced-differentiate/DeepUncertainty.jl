@testset "Trainable Distributions" begin
    l = TrainableDistribution((2, 3)) |> gpu
    grads = gradient(params(l)) do
        weight = l()
        kl = KLDivergence(l)
        return kl
    end
    for param in params(l)
        @test size(param) == size(grads[param])
        @test grads[param] isa CuArray
    end
end
