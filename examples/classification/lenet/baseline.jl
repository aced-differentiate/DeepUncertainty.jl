using Flux
using Flux.Data:DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, glorot_normal, label_smoothing
using Flux.Losses:logitcrossentropy
using Statistics, Random
using Logging:with_logger
using ProgressMeter:@showprogress
import MLDatasets
using CUDA
using Formatting

using DeepUncertainty
include("utils.jl")

function LeNet5(args; imgsize=(28, 28, 1), nclasses=10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        Conv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses),
    )
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(args, loader, model, device)
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

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 256      # batch size
    epochs = 50          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1      # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    dropout = 0.1
    sample_size = 10
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
    train_loader, test_loader, ood_train_loader, ood_test_loader = get_data(args)
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
        @info "Test metrics"
        eval_loss_accuracy(args, test_loader, model, device)
        @info "Test OOD metrics"
        eval_loss_accuracy(args, ood_test_loader, model, device)
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch = 1:args.epochs
        @info "Epoch: $(epoch)"
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                total_loss = loss(ŷ, y)
                return total_loss
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
