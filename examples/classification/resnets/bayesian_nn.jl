using Flux
using Flux: onehotbatch, onecold, flatten, Zygote
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using ProgressMeter: @showprogress
using Formatting

using DeepUncertainty
include("utils.jl")
include("models/bayesian_nn.jl")

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 64
    lr::Float64 = 0.1
    epochs::Int = 50
    valsplit::Float64 = 0.1
    sample_size = 10
    complexity_constant = 1e-8
end

function accuracy(preds, labels)
    acc = sum(onecold(preds |> cpu) .== onecold(labels |> cpu))
    return acc
end

function test(args, loader, model)
    loss = 0
    acc = 0
    ece = 0
    entropy = 0
    ntot = 0
    for (x, y) in loader
        predictions = []
        x, y = x |> gpu, y |> gpu
        logits = model(x)

        # Loop through each model's predictions 
        for ensemble = 1:args.sample_size
            logits = model(x)
            push!(predictions, logits)
        end

        # Get the mean predictions
        mean_preds = Flux.batch(predictions)
        mean_predictions = mean(mean_preds, dims = ndims(mean_preds))
        logits = dropdims(mean_predictions, dims = ndims(mean_predictions))

        n = size(logits)[end]
        loss += logitcrossentropy(logits, y) * n
        acc += accuracy(logits, y)
        ece += expected_calibration_error(logits |> cpu, onecold(y |> cpu)) * n
        entropy += mean(calculate_entropy(logits |> cpu)) * n
        ntot += n
    end

    mean_loss = loss / ntot |> round4
    mean_acc = acc / ntot * 100 |> round4
    mean_ece = ece / ntot |> round4
    mean_entropy = entropy / ntot |> round4

    @info (format(
        "Mean Loss: {} Mean Accuracy: {} Mean ECE: {} Mean Entropy: {}",
        mean_loss,
        mean_acc,
        mean_ece,
        mean_entropy,
    ))
    @info "==========================================================="
    return nothing
end


function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)

    # Load the train, validation data 
    train_loader, test_loader = get_data(args)

    @info("Constructing Model")
    m = VariationalResNet18(nclasses = 10) |> gpu

    ## Training
    # Defining the optimizer
    opt = Nesterov(args.lr)
    ps = Flux.params(m)

    test(args, test_loader, m)

    @info("Training....")
    # Starting to train models
    for epoch = 1:args.epochs
        @info "Epoch $epoch"

        function loss_fn(x, y)
            l = logitcrossentropy(m(x), y)
            layers = Zygote.@ignore Flux.modules(m)
            l += args.complexity_constant * sum(normal_kl_divergence.(layers))
        end

        @showprogress for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            total_loss = 0
            gs = Flux.gradient(ps) do
                for sample in args.sample_size
                    total_loss += loss_fn(x, y)
                end
                total_loss /= args.sample_size
            end
            Flux.update!(opt, ps, gs)
        end
        test(args, test_loader, m)
    end
    test(args, test_loader, m)

    return m
end

m = train()
