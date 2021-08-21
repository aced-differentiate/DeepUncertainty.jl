using Test 
using Flux 
using DeepUncertainty:ConvBatchEnsemble

@testset "ConvBatchEnsemble" begin   
    input = rand(Float32, (100, 100, 8, 32))
    layer = ConvBatchEnsemble((5, 5), 8 => 16, 1, 4, relu)
    output = layer(input) 
    @test size(output) == (96, 96, 16, 32)
end 