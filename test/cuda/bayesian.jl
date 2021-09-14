@testset "Trainable Distributions" begin
    @testset "Trainable MvNormal" begin
        l = TrainableMvNormal((2, 3)) |> gpu
        grads = gradient(params(l)) do
            weight = l()
            kl = NormalKLDivergence(l)
            return kl
        end
        for param in params(l)
            @test size(param) == size(grads[param])
            @test grads[param] isa CuArray
        end
    end
end
