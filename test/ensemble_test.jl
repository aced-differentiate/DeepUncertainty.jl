using Flux
using Test 
using Random
using Formatting
using Flux, Statistics
using Flux.Data:DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses:logitcrossentropy
using Base:@kwdef
using MLDatasets
using BenchmarkTools 
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
include("../src/ensemble.jl")

# arguments for the `train` function 
@kwdef mutable struct Args
    η::Float64 = 3e-4           # learning rate
    batchsize::Int = 100        # batch size
    epochs::Int = 1             # number of epochs
    use_cuda::Bool = true       # use gpu (if cuda available)
    seed = 0                    # set seed > 0 for reproducibility
    infotime = 1 	            # report every `infotime` epochs
    checktime = 1               # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true             # log training with tensorboard
    savepath = "runs"           # results path
    model_name = "simple_mlp"   # model name
    ensemble_size::Int = 3      # Ensemble size 
end

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reduce dataset size for testing 
    xtrain, ytrain = xtrain[:, :, 1:1000], ytrain[1:1000]
    xtest, ytest = xtest[:, :, 1:1000], ytest[1:1000]
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

function build_model(; imgsize=(28, 28, 1), nclasses=10)
    return Chain(
 	        Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end

function loss_and_accuracy(data_loader, model)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(cpu(ŷ)) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return ls / num, acc / num
end


function train(args::Args, savename, savedir)
    # Create test and train dataloaders
    train_loader, test_loader = getdata(args)

    # Construct model
    model = build_model()
    ps = Flux.params(model) # model's trainable parameters
    # Check if model directory is created 
    !ispath(savedir) && mkpath(savedir)
    modelpath = joinpath(savedir, savename) 
    
    ## Optimizer
    opt = ADAM(args.η)

    # TB logger
    if args.tblogger 
        tblogger = TBLogger(savedir, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        # @info "TensorBoard logging at \"$(savedir)\""
        @info format("TensorBoard logging at {}", savedir)
    end
    
    training_complete = false 
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        
        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model)
        test_loss, test_acc = loss_and_accuracy(test_loader, model)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        
        # Save the model
        BSON.@save modelpath model
        # @info "\"$(model_name))\" at epoch  \"$(epoch)\" saved in \"$(model_dir)\""
        @info format("Model at epoch {} saved.", epoch)
    end

    # Mark training as done 
    training_complete = true 
    @info format("Finished training")
    # Save the model and epoch and if training is done -- to resume 
    BSON.@save modelpath model

    return nothing 
end

@testset "Ensemble Train" begin
    args = Args() # collect options in a struct for convenience
    ensemble_train(args, train)
end 