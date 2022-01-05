using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using ProgressMeter: @showprogress
using Formatting
using MLDatasets: CIFAR10, SVHN2

using DeepUncertainty

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 64
    lr::Float64 = 3e-4
    epochs::Int = 50
    valsplit::Float64 = 0.1
    sample_size = 10
    complexity_constant = 1e-8
end

function accuracy(preds, labels)
    acc = sum(onecold(preds |> cpu) .== onecold(labels |> cpu))
    return acc
end


loss(ŷ, y) = logitcrossentropy(ŷ, y)

num_params(model) = sum(length, Flux.params(model))

round4(x) = round(x, digits = 4)

function get_data(args)
    x, y = CIFAR10.traindata()
    train_x = float(x)
    train_y = onehotbatch(y, 0:9)
    train_data = (train_x, train_y)

    test_x, test_y = CIFAR10.testdata()
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    test_data = (test_x, test_y)

    # # OOD dataset
    # ood_test_x, ood_test_y = SVHN2.testdata() 
    # ood_test_x = float(ood_test_x)
    # ood_test_y = onehotbatch(ood_test_y, 1:10)

    train_loader =
        DataLoader(train_data, batchsize = args.batchsize, shuffle = true, partial = false)
    test_loader = DataLoader(test_data, batchsize = args.batchsize)
    #ood_test_loader = DataLoader(ood_test_data, batchsize=args.batchsize)

    return train_loader, test_loader
end

# VGG16 and VGG19 models
function vgg16()
    dropout = 0.2
    Chain(
        MCConv((3, 3), 3 => 64, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        MCConv((3, 3), 64 => 64, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        MaxPool((2, 2)),
        MCConv((3, 3), 64 => 128, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        MCConv((3, 3), 128 => 128, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        MaxPool((2, 2)),
        MCConv((3, 3), 128 => 256, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        MCConv((3, 3), 256 => 256, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        MCConv((3, 3), 256 => 256, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        MaxPool((2, 2)),
        MCConv((3, 3), 256 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MCConv((3, 3), 512 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MCConv((3, 3), 512 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MaxPool((2, 2)),
        MCConv((3, 3), 512 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MCConv((3, 3), 512 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MCConv((3, 3), 512 => 512, dropout, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MaxPool((2, 2)),
        flatten,
        Dense(512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 10),
    )
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
        ece += expected_calibration_error(logits, onecold(y)) * n
        entropy += mean(calculate_entropy(logits)) * n
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
    m = vgg16() |> gpu

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    test(args, test_loader, m)

    @info("Training....")
    # Starting to train models
    for epoch = 1:args.epochs
        @info "Epoch $epoch"

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
