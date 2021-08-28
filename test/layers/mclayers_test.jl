using Test
using DeepUncertainty: MCDense, MCConv

@testset "MC Dense" begin
    dropout_rate = 0.35
    # Test MC Dense layer 
    a = rand(Float32, 8, 32)
    layer = MCDense(8, 16, dropout_rate)
    output = layer(a)
    number_of_zeros = count(x -> (x == 0.0), output)
    sparsity = number_of_zeros / sum(length, output)
    @test isequal(size(output), (16, 32))
    @test isapprox(dropout_rate, sparsity; atol = 0.05)

    # Test MC dense dropout toggle 
    output = layer(a, dropout = false)
    number_of_zeros = count(x -> (x == 0.0), output)
    sparsity = number_of_zeros / sum(length, output)
    @test isapprox(0, sparsity; atol = 0.05)
end

@testset "MC Conv" begin
    dropout_rate = 0.4
    # Test MC conv layer 
    a = rand(Float32, 32, 32, 3, 32)
    layer = MCConv((5, 5), 3 => 6, dropout_rate)
    output = layer(a)
    number_of_zeros = count(x -> (x == 0.0), output)
    sparsity = number_of_zeros / sum(length, output)
    # Test the output shape 
    @test isequal(size(output), (28, 28, 6, 32))
    # Test the sparsity percentage in the array 
    @test isapprox(dropout_rate, sparsity; atol = 0.05)

    # Test MC conv dropout toggle 
    output = layer(a, dropout = false)
    number_of_zeros = count(x -> (x == 0.0), output)
    sparsity = number_of_zeros / sum(length, output)
    @test isapprox(0, sparsity; atol = 0.05)
end
