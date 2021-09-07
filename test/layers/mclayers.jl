function test_sparsity(x, target_sparsity; atol = 0.05)
    number_of_zeros = count(ele -> (ele == 0.0), x)
    sparsity = number_of_zeros / sum(length, x)
    @test isapprox(target_sparsity, sparsity; atol)
end

@testset "MC Dense" begin
    dropout_rate = 0.35
    # Test MC Dense layer 
    a = rand(Float32, 8, 32)
    layer = MCDense(8, 16, dropout_rate)
    output = layer(a)
    @test isequal(size(output), (16, 32))
    test_sparsity(output, dropout_rate)
    # Test MC dense dropout toggle 
    output = layer(a, dropout = false)
    test_sparsity(output, 0)
end

@testset "MC Conv" begin
    dropout_rate = 0.4
    # Test MC conv layer 
    a = rand(Float32, 32, 32, 3, 32)
    layer = MCConv((5, 5), 3 => 6, dropout_rate)
    output = layer(a)
    # Test the output shape 
    @test isequal(size(output), (28, 28, 6, 32))
    # Test the sparsity percentage in the array 
    test_sparsity(output, dropout_rate)
    # Test MC conv dropout toggle 
    output = layer(a, dropout = false)
    test_sparsity(output, 0)
end
