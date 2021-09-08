using DeepUncertainty
using Test
using Flux
using Flux.CUDA
using Flux: cpu, gpu

@testset "Layers" begin
    # MC layers 
    include("./layers/mclayers.jl")
    # Batch ensembe layers 
    include("./layers/batchensemble.jl")
end

@testset "CUDA" begin
    if CUDA.functional()
        include("cuda/runtests.jl")
    else
        @warn "CUDA unavailable, not testing GPU support"
    end
end
