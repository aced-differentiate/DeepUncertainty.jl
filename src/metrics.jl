using Statistics

function normalise(x::AbstractArray; dims=ndims(x))
    μ = mean(x, dims=dims)
    σ = std(x, dims=dims, corrected=false)
    return μ, σ
end

input = rand(10, 2, 4)
mu, sigma = normalise(input, dims=3)
println(mu)
println(sigma)