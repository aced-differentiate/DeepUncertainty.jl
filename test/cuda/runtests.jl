using Flux, Test, CUDA

@info "Testing GPU Support"
CUDA.allowscalar(false)

include("layers/mclayers_gpu.jl")
include("layers/batchensemble_gpu.jl")
