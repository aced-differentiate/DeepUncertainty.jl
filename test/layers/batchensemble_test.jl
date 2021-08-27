using Test
using Flux

include("../../src/layers/batchensemble/dense.jl")
include("../../src/layers/batchensemble/conv.jl")

@testset "DenseBatchEnsemble" begin
    ensemble_size = 3
    samples_per_model = 4
    input_dim = 5
    output_dim = 5
    rank = 1
    inputs = rand(input_dim, samples_per_model)
    layer = DenseBatchEnsemble(
        input_dim,
        output_dim,
        rank,
        ensemble_size;
        alpha_init = ones,
        gamma_init = ones,
    )
    batch_inputs = repeat(inputs, 1, ensemble_size)
    batch_outputs = layer(batch_inputs)
    # Do the computation in for loop to compare outputs 
    loop_outputs = []
    for i = 1:ensemble_size
        perturbed_inputs = inputs .* layer.alpha[i]
        outputs = layer.layer(perturbed_inputs) .* layer.gamma[i]
        outputs = layer.ensemble_act.(outputs .+ layer.ensemble_bias[i])
        push!(loop_outputs, outputs)
    end
    loop_outputs = Flux.batch(loop_outputs)
    loop_outputs = reshape(loop_outputs, (output_dim, samples_per_model * ensemble_size))
    @test size(batch_outputs) == size(loop_outputs)
    @test isapprox(batch_outputs, loop_outputs)
end


@testset "ConvBatchEnsemble" begin
    ensemble_size = 3
    samples_per_model = 4
    input_dim = 5
    output_dim = 10
    rank = 1
    inputs = rand(Float32, 10, 10, input_dim, samples_per_model)
    layer = ConvBatchEnsemble((5, 5), 5 => 10, 1, 4, relu)
    batch_inputs = repeat(inputs, 1, 1, 1, ensemble_size)
    batch_outputs = layer(batch_inputs)

    # Do the computation in for loop to compare outputs 
    loop_outputs = []
    for i = 1:ensemble_size
        perturbed_inputs = inputs .* layer.alpha[i]
        outputs = layer.layer(perturbed_inputs) .* layer.gamma[i]
        outputs = layer.ensemble_act.(outputs .+ layer.ensemble_bias[i])
        push!(loop_outputs, outputs)
    end
    loop_outputs = Flux.batch(loop_outputs)
    loop_outputs_size = size(loop_outputs)
    loop_outputs = reshape(
        loop_outputs,
        (
            loop_outputs_size[1],
            loop_outputs_size[2],
            output_dim,
            samples_per_model * ensemble_size,
        ),
    )
    @test size(batch_outputs) == size(loop_outputs)
    @test isapprox(batch_outputs, loop_outputs)
end
