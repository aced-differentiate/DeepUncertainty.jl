@testset "Trainable Distributions" begin
    @testset "Trainable MvNormal" begin
        l = TrainableMvNormal((2, 3))
        grads = gradient(params(l)) do
            weight = l()
            kl = normal_kl_divergence(l)
            return kl
        end
        for param in params(l)
            @test size(param) == size(grads[param])
        end

        grads = gradient(params(l)) do
            weight = l()
            kl = scale_mixture_kl_divergence(l)
            return kl
        end
        for param in params(l)
            @test size(param) == size(grads[param])
        end
    end
end
