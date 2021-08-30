using DeepUncertainty
using Test

@testset "Layers" begin
    # MC layers 
    include("./layers/mclayers_test.jl")
    # Batch ensembe layers 
    include("./layers/batchensemble_test.jl")
end
