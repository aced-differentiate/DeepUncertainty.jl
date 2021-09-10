@testset "Trainable Distributions" begin
    l = TrainableDistribution((2, 3))
    grads = gradient(params(l)) do
        weight = l()
        kl = KLDivergence(l)
        return kl
    end
    for param in params(l)
        @test size(param) == size(grads[param])
    end
end
