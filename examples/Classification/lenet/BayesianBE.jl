using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, glorot_normal, label_smoothing
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using ProgressMeter: @showprogress
import MLDatasets
using CUDA
using Formatting

using DeepUncertainty
include("utils.jl")

function LeNet5(args; imgsize = (28, 28, 1), nclasses = 10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        VariationalConvBE((5, 5), imgsize[end] => 6, args.rank, args.ensemble_size, relu),
        MaxPool((2, 2)),
        VariationalConvBE((5, 5), 6 => 16, args.rank, args.ensemble_size, relu),
        MaxPool((2, 2)),
        flatten,
        VariationalDenseBE(prod(out_conv_size), 120, args.rank, args.ensemble_size, relu),
        VariationalDenseBE(120, 84, args.rank, args.ensemble_size, relu),
        VariationalDenseBE(84, nclasses, args.rank, args.ensemble_size),
    )
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 64      # batch size
    epochs = 10          # number of epochs
    use_cuda = true      # if true use cuda (if available)
    infotime = 1      # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    rank = 1
    ensemble_size = 4
    sample_size = 10
    complexity_constant = 1e-8
end

function kldivergence(model)
    loss = 0
    modules = Flux.modules(model)
    for layer in modules
        if layer isa AbstractTrainableDist
            loss += NormalKLDivergence(layer)
        end
    end
    return loss
end

function train(; kws...)
    args = Args(; kws...)
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
        ensembles_evaluation(args, test_loader, model, device)
        @info "Test OOD metrics"
        ensembles_evaluation(args, ood_test_loader, model, device)
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
                total_loss = 0
                for sample in args.sample_size
                    ŷ = model(x)
                    total_loss += loss(ŷ, y)
                    kl_loss = sum(normal_kl_divergence.(Flux.modules(model)))
                    total_loss += args.complexity_constant * kl_loss
                end
                total_loss /= args.sample_size
                return total_loss
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
