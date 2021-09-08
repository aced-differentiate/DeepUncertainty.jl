## Classification of MNIST dataset 
## with the convolutional neural network known as LeNet5.
## This script also combines various
## packages from the Julia ecosystem with Flux.
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, glorot_normal, label_smoothing
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA
using Formatting

using DeepUncertainty

# LeNet5 "constructor". 
# The model can be adapted to any image size
# and any number of output classes.
function LeNet5(args; imgsize = (28, 28, 1), nclasses = 10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        ConvBatchEnsemble((5, 5), imgsize[end] => 6, args.rank, args.ensemble_size, relu),
        MaxPool((2, 2)),
        ConvBatchEnsemble((5, 5), 6 => 16, args.rank, args.ensemble_size, relu),
        MaxPool((2, 2)),
        flatten,
        DenseBatchEnsemble(prod(out_conv_size), 120, args.rank, args.ensemble_size, relu),
        DenseBatchEnsemble(120, 84, args.rank, args.ensemble_size, relu),
        DenseBatchEnsemble(84, nclasses, args.rank, args.ensemble_size),
    )
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader(
        (xtrain, ytrain),
        batchsize = args.batchsize,
        shuffle = true,
        partial = false,
    )
    test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize, partial = false)

    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function accuracy(preds, labels)
    acc = sum(onecold(preds |> cpu) .== onecold(labels |> cpu))
    return acc
end

function eval_loss_accuracy(args, loader, model, device)
    l = [0.0f0 for x = 1:args.ensemble_size]
    acc = [0 for x = 1:args.ensemble_size]
    ece_list = [0.0f0 for x = 1:args.ensemble_size]
    ntot = 0
    mean_l = 0
    mean_acc = 0
    mean_ece = 0
    for (x, y) in loader
        x = repeat(x, 1, 1, 1, args.ensemble_size)
        x, y = x |> device, y |> device
        # Perform the forward pass 
        ŷ = model(x)
        ŷ = softmax(ŷ, dims = 1)
        # Reshape the predictions into [classes, batch_size, ensemble_size
        reshaped_ŷ = reshape(ŷ, size(ŷ)[1], args.batchsize, args.ensemble_size)
        # Loop through each model's predictions 
        for ensemble = 1:args.ensemble_size
            model_predictions = reshaped_ŷ[:, :, ensemble]
            # Calculate individual loss 
            l[ensemble] += loss(model_predictions, y) * size(model_predictions)[end]
            acc[ensemble] += accuracy(model_predictions, y)
            ece_list[ensemble] +=
                ExpectedCalibrationError(model_predictions |> cpu, onecold(y |> cpu)) *
                args.batchsize
        end
        # Get the mean predictions
        mean_predictions = mean(reshaped_ŷ, dims = ndims(reshaped_ŷ))
        mean_predictions = dropdims(mean_predictions, dims = ndims(mean_predictions))
        mean_l += loss(mean_predictions, y) * size(mean_predictions)[end]
        mean_acc += accuracy(mean_predictions, y)
        mean_ece +=
            ExpectedCalibrationError(mean_predictions |> cpu, onecold(y |> cpu)) *
            args.batchsize
        ntot += size(mean_predictions)[end]
    end
    # Normalize the loss 
    losses = [loss / ntot |> round4 for loss in l]
    acc = [a / ntot * 100 |> round4 for a in acc]
    ece_list = [x / ntot |> round4 for x in ece_list]
    # Calculate mean loss 
    mean_l = mean_l / ntot |> round4
    mean_acc = mean_acc / ntot * 100 |> round4
    mean_ece = mean_ece / ntot |> round4

    # Print the per ensemble mode loss and accuracy 
    for ensemble = 1:args.ensemble_size
        @info (format(
            "Model {} Loss: {} Accuracy: {} ECE: {}",
            ensemble,
            losses[ensemble],
            acc[ensemble],
            ece_list[ensemble],
        ))
    end
    @info (format(
        "Mean Loss: {} Mean Accuracy: {} Mean ECE: {}",
        mean_l,
        mean_acc,
        mean_ece,
    ))
    @info "==========================================================="
    return nothing
end

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits = 4)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 32      # batch size
    epochs = 10          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1      # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    savepath = "runs/"    # results path
    rank = 1
    ensemble_size = 4
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5(args) |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"

    ps = Flux.params(model)

    opt = ADAM(args.η)
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end

    function report(epoch)
        # @info "Train Metrics"
        # eval_loss_accuracy(args, train_loader, model, device)
        @info "Test metrics"
        eval_loss_accuracy(args, test_loader, model, device)
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch = 1:args.epochs
        @showprogress for (x, y) in train_loader
            # Make copies of batches for ensembles 
            x = repeat(x, 1, 1, 1, args.ensemble_size)
            y = repeat(y, 1, args.ensemble_size)
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
