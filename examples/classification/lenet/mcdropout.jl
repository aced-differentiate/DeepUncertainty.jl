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

# function LeNet5(args; imgsize=(28, 28, 1), nclasses=10)
#     out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

#     return Chain(
#         MCConv((5, 5), imgsize[end] => 6, args.dropout, relu),
#         MaxPool((2, 2)),
#         MCConv((5, 5), 6 => 16, args.dropout, relu),
#         MaxPool((2, 2)),
#         flatten,
#         MCDense(prod(out_conv_size), 120, args.dropout, relu),
#         MCDense(120, 84, args.dropout, relu),
#         MCDense(84, nclasses, args.dropout),
#     )
# end

function LeNet5(args; imgsize=(28, 28, 1), nclasses=10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)
    dropout = (x; k...) -> Flux.dropout(x, args.dropout; k...)

    return Chain(
        MCLayer(Conv((5, 5), imgsize[end] => 6, relu), dropout),
        MaxPool((2, 2)),
        MCLayer(Conv((5, 5), 6 => 16, relu), dropout),
        MaxPool((2, 2)),
        flatten,
        MCLayer(Dense(prod(out_conv_size), 120, relu), dropout),
        MCLayer(Dense(120, 84, relu), dropout),
        MCLayer(Dense(84, nclasses), dropout),
    )
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 256      # batch size
    epochs = 100          # number of epochs
    use_cuda = true      # if true use cuda (if available)
    infotime = 1        # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    dropout = 0.2
    sample_size = 25
    seed = 0  
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
        monte_carlo_evaluation(args, test_loader, model, device)
        @info "Test OOD metrics"
        monte_carlo_evaluation(args, ood_test_loader, model, device)
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
                loss(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
