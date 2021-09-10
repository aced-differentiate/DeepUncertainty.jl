using Flux
using Flux: gpu, params
using DeepUncertainty

l = TrainableDistribution((2, 3)) |> gpu
grads = gradient(params(l)) do
    weight = l()
    kl = KLDivergence(l)
    return kl
end
for param in params(l)
    println(grads[param])
end
