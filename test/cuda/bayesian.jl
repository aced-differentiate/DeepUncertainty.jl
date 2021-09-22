function calc_kld(mean, stddev, sample)
    posterior = DistributionsAD.TuringMvNormal(mean, abs.(stddev))
    posterior_loglikelihood = DistributionsAD.loglikelihood(posterior, sample)

    # Calculate prior loglikelihood 
    μ2 = gpu(zeros(size(mean)[1]))
    σ2 = gpu(ones(size(mean)[1]))
    prior = DistributionsAD.TuringMvNormal(μ2, σ2)
    prior_loglikelihood = DistributionsAD.loglikelihood(prior, sample)

    return (posterior_loglikelihood - prior_loglikelihood)
end


@testset "Trainable Distributions" begin
    @testset "Trainable MvNormal Normal KL divergence" begin
        mean =
            Float32.([
                1.3070704,
                0.21997994,
                0.045152083,
                -0.90033674,
                0.27203867,
                0.017634762,
            ])
        stddev =
            Float32.([
                0.18203342,
                0.5463153,
                1.3376112,
                -0.19215275,
                0.5753563,
                -0.19471616,
            ])
        sample =
            Float32.([
                1.0493519,
                -0.14344203,
                0.27405795,
                -1.0525246,
                0.2674973,
                -0.3156111,
            ])

        mean = gpu(mean)
        stddev = gpu(stddev)
        sample = gpu(sample)
        base_grads = gradient(params(mean, stddev)) do
            kl = calc_kld(mean, stddev, sample)
            return kl
        end

        l = TrainableMvNormal((2, 3)) |> gpu
        Flux.Zygote.@ignore copyto!(l.mean, mean)
        Flux.Zygote.@ignore copyto!(l.stddev, stddev)
        Flux.Zygote.@ignore copyto!(l.sample, sample)
        struct_grads = gradient(params(l)) do
            kl = normal_kl_divergence(l)
            return kl
        end
        for param in zip(params(l), params(mean, stddev))
            @test isapprox(struct_grads[param[1]], base_grads[param[2]])
            @test size(param[1]) == size(struct_grads[param[1]])
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
