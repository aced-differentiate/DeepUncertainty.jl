@testset "Trainable Distributions" begin
    @testset "Trainable MvNormal Normal KL divergence" begin
        l = TrainableMvNormal((2, 3)) |> gpu
        grads = gradient(params(l)) do
            weight = l()
            kl = normal_kl_divergence(l)
            return kl
        end
        for param in params(l)
            @test size(param) == size(grads[param])
            @test grads[param] isa CuArray
        end
    end

    @testset "Trainable MvNormal Scale Mixture KL" begin
        l = TrainableMvNormal((2, 3)) |> gpu
        grads = gradient(params(l)) do
            weight = l()
            kl = scale_mixture_kl_divergence(l)
            return kl
        end
        for param in params(l)
            @test size(param) == size(grads[param])
            @test grads[param] isa CuArray
        end
    end
end
