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
        VariationalConv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        VariationalConv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        flatten,
        VariationalDense(prod(out_conv_size), 120, relu),
        VariationalDense(120, 84, relu),
        VariationalDense(84, nclasses),
    )
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 0.1            # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 256      # batch size
    epochs = 100          # number of epochs
    use_cuda = true      # if true use cuda (if available)
    infotime = 1      # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    dropout = 0.1
    sample_size = 10
    complexity_constant = 1e-8
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

    opt = Nesterov(args.η)

    function report(epoch)
        @info "Test metrics"
        monte_carlo_evaluation(args, test_loader, model, device)
        @info "Test OOD metrics"
        monte_carlo_evaluation(args, ood_test_loader, model, device)
    end

    ## TRAINING
    @info "Start Training"
    for epoch = 1:args.epochs
        @showprogress for (x, y) in train_loader
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
