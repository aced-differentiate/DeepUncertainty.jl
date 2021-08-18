using Test 
using Flux 
using DeepUncertainty:DenseBatchEnsemble

@testset "DenseBatchEnsemble" begin   
    input = rand(Float32, (8, 32))
    layer = DenseBatchEnsemble(8, 16, 1, 4)
    output = layer(input) 
    @test size(output) == (16, 32)
end 