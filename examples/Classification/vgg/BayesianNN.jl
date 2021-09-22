using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLDataPattern: splitobs
using ProgressMeter: @showprogress

using DeepUncertainty

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function get_processed_data(args)
    x, y = CIFAR10.traindata()

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at = 1 - args.valsplit)

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

# VGG16 and VGG19 models
function vgg16()
    Chain(
        VariationalConv((3, 3), 3 => 64, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        VariationalConv((3, 3), 64 => 64, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        MaxPool((2, 2)),
        VariationalConv((3, 3), 64 => 128, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        VariationalConv((3, 3), 128 => 128, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        MaxPool((2, 2)),
        VariationalConv((3, 3), 128 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        VariationalConv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        VariationalConv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        MaxPool((2, 2)),
        VariationalConv((3, 3), 256 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        VariationalConv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        VariationalConv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MaxPool((2, 2)),
        VariationalConv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        VariationalConv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        VariationalConv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
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

@with_kw mutable struct Args
    batchsize::Int = 64
    lr::Float64 = 3e-4
    epochs::Int = 50
    valsplit::Float64 = 0.1
    sample_size = 3
    complexity_constant = 1e-8
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function accuracy(preds, labels)
    acc = sum(onecold(preds |> cpu) .== onecold(labels |> cpu))
    return acc
end

function test(args, loader, model)
    l = [0.0f0 for x = 1:args.sample_size]
    acc = [0 for x = 1:args.sample_size]
    ece_list = [0.0f0 for x = 1:args.sample_size]
    entropies = [0.0f0 for x = 1:args.sample_size]
    ntot = 0
    mean_l = 0
    mean_acc = 0
    mean_ece = 0
    mean_entropy = 0
    for (x, y) in loader
        predictions = []
        x, y = x |> gpu, y |> gpu

        # Loop through each model's predictions 
        for ensemble = 1:args.sample_size
            model_predictions = model(x)
            model_predictions = softmax(model_predictions, dims = 1)
            push!(predictions, model_predictions)
            # Calculate individual loss 
            l[ensemble] += loss(model_predictions, y) * size(model_predictions)[end]
            acc[ensemble] += accuracy(model_predictions, y)
            ece_list[ensemble] +=
                expected_calibration_error(model_predictions |> cpu, onecold(y |> cpu)) *
                args.batchsize
            entropies[ensemble] +=
                mean(calculate_entropy(model_predictions |> cpu)) * args.batchsize
        end
        # Get the mean predictions
        predictions = Flux.batch(predictions)
        mean_predictions = mean(predictions, dims = ndims(predictions))
        mean_predictions = dropdims(mean_predictions, dims = ndims(mean_predictions))
        mean_l += loss(mean_predictions, y) * size(mean_predictions)[end]
        mean_acc += accuracy(mean_predictions, y)
        mean_ece +=
            expected_calibration_error(mean_predictions |> cpu, onecold(y |> cpu)) *
            args.batchsize
        mean_entropy += mean(calculate_entropy(mean_predictions |> cpu)) * args.batchsize
        ntot += size(mean_predictions)[end]
    end
    # Normalize the loss 
    losses = [loss / ntot |> round4 for loss in l]
    acc = [a / ntot * 100 |> round4 for a in acc]
    ece_list = [x / ntot |> round4 for x in ece_list]
    entropies = [x / ntot |> round4 for x in entropies]
    # Calculate mean loss 
    mean_l = mean_l / ntot |> round4
    mean_acc = mean_acc / ntot * 100 |> round4
    mean_ece = mean_ece / ntot |> round4
    mean_entropy = mean_entropy / ntot |> round4

    # Print the per ensemble mode loss and accuracy 
    for ensemble = 1:args.sample_size
        @info (format(
            "Sample {} Loss: {} Accuracy: {} ECE: {} Entropy: {}",
            ensemble,
            losses[ensemble],
            acc[ensemble],
            ece_list[ensemble],
            entropies[ensemble],
        ))
    end
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


function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)

    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)

    train_loader = DataLoader(train_data, batchsize = args.batchsize, shuffle = true)
    val_loader = DataLoader(val_data, batchsize = args.batchsize)

    @info("Constructing Model")
    m = vgg16() |> gpu

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch = 1:args.epochs
        @info "Epoch $epoch"

        @showprogress for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(ps) do
                total_loss = 0
                for sample in args.sample_size
                    ŷ = m(x)
                    total_loss += loss(ŷ, y)
                    kl_loss = sum(scale_mixture_kl_divergence.(Flux.modules(m)))
                    total_loss += args.complexity_constant * kl_loss
                end
                total_loss /= args.sample_size
                return total_loss
            end
            Flux.update!(opt, ps, gs)
        end
        test(args, val_loader, m)
    end

    return m
end

m = train()
test(args, val_loader, m)
