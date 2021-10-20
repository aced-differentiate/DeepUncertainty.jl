using Flux, ParameterSchedulers
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using ProgressMeter: @showprogress
using Formatting
using ParameterSchedulers: Scheduler

using DeepUncertainty
include("utils.jl")
include("models/resnets.jl")

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 100
    lr::Float64 = 0.1
    epochs::Int = 200
    valsplit::Float64 = 0.1
    sample_size = 3
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
        x, y = x |> gpu, y |> gpu
        logits = model(x)

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
    m = ResNet18(nclasses = 10) |> gpu

    ## Training
    # Defining the optimizer
    # opt = Nesterov(args.lr)
    opt =
        Scheduler(Step(λ = 1.0, γ = 0.8, step_sizes = [500, 1000, 1500]), Nesterov(args.lr))
    ps = Flux.params(m)

    test(args, test_loader, m)

    @info("Training....")

    # Starting to train models
    for epoch = 1:args.epochs
        @info (format("Epoch: {} Learning rate: {}", epoch, opt.optim.eta))

        loss_fn(x, y) = logitcrossentropy(m(x), y)

        @showprogress for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(ps) do
                loss_fn(x, y)
            end
            Flux.update!(opt, ps, gs)
        end
        test(args, test_loader, m)
    end
    test(args, test_loader, m)

    return m
end

m = train()
