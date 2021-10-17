using CUDA:include
using Flux, ParameterSchedulers
using Flux: onehotbatch, onecold, flatten
using Flux.Losses:logitcrossentropy
using Flux.Data:DataLoader
using Parameters:@with_kw
using Statistics:mean
using CUDA
using MLDatasets:CIFAR10
using MLDataPattern:splitobs
using ProgressMeter:@showprogress

include("models/resnets.jl")

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

device = gpu

function eval_loss_accuracy(loader, model, device)
    ntot = 0
    mean_l = 0
    mean_acc = 0
    mean_ece = 0
    mean_entropy = predictions = []
    targets = []
    for (x, y) in loader
        x, y = x |> device, y |> device
        model_preds = model(x)
        push!(predictions, cpu(model_preds))
        push!(targets, cpu(y))
        mean_l += loss(model_preds, y) * size(model_preds)[end]
        ntot += size(model_preds)[end]
    end
    # Calculate mean loss 
    mean_l = mean_l / ntot |> round4
    predictions = Flux.batch(predictions)
    targets = Flux.batch(targets)
    pred_shape = size(predictions)
    target_shape = size(targets)
    predictions = reshape(predictions, (pred_shape[1], pred_shape[2] * pred_shape[3]))
    targets = reshape(targets, (target_shape[1], target_shape[2] * target_shape[3]))
    mean_acc = accuracy(predictions, targets) / ntot * 100 |> round4
    mean_entropy = round4(mean(calculate_entropy(predictions)))
    mean_ece = round4(expected_calibration_error(predictions, onecold(targets)))

    @info (format(
        "Mean Loss: {} Mean Accuracy: {} Mean ECE: {} Mean Entropy: {}",
        mean_l,
        mean_acc,
        mean_ece,
        mean_entropy,
    ))
    @info "==========================================================="
    return nothing
end


function get_processed_data(args)
    x, y = CIFAR10.traindata()

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1 - args.valsplit)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)

    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10.testdata()

    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)

    return test_x, test_y
end

@with_kw mutable struct Args
    batchsize::Int = 128
    lr::Float64 = 0.1
    epochs::Int = 200
    valsplit::Float64 = 0.1

end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)

    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)

    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")
    m = ResNet18(nclasses=10) |> device

    loss(x, y) = logitcrossentropy(m(x), y)

    ## Training
    # Defining the optimizer
    opt = Nesterov(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch = 1:args.epochs
        @info "Epoch $epoch"

        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)
        end

        eval_loss_accuracy(val_loader, m, device)
    end

    return m
end

m = train()
test_data = get_test_data()
test_loader = DataLoader(test_data, batchsize=args.batchsize)
eval_loss_accuracy(test_loader, m, device)
