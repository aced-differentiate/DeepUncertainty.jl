using Flux
using BSON
using Flux: gradient
using Flux.Optimise: update!
using DelimitedFiles, Statistics
using Parameters: @with_kw

using Formatting

include("../src/ensemble.jl")

# This replicates the housing data example from the Knet.jl readme. Although we
# could have reused more of Flux (see the mnist example), the library's
# abstractions are very lightweight and don't force you into any particular
# strategy.

# Struct to define hyperparameters
@with_kw mutable struct Hyperparams
    lr::Float64 = 0.1# learning rate
    split_ratio::Float64 = 0.1# Train Test split ratio, define percentage of data to be used as Test data
    savepath::String = "runs"
    model_name::String = "housing_predictor"
    ensemble_size::Int = 3
end

function get_processed_data(args)
    isfile("housing.data") || download(
        "https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
        "housing.data",
    )

    rawdata = readdlm("housing.data")'

    # The last feature is our target -- the price of the house.
    split_ratio = args.split_ratio # For the train test split

    x = rawdata[1:13, :]
    y = rawdata[14:14, :]

    # Normalise the data
    x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

    # Split into train and test sets
    split_index = floor(Int, size(x, 2) * split_ratio)
    x_train = x[:, 1:split_index]
    y_train = y[:, 1:split_index]
    x_test = x[:, split_index+1:size(x, 2)]
    y_test = y[:, split_index+1:size(x, 2)]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data, test_data
end

# Struct to define model
mutable struct model
    W::AbstractArray
    b::AbstractVector
end

# Function to predict output from given parameters
function predict(model; x)
    return model.W * x .+ model.b
end

# Define the mean squared error function to be used in the loss 
# function. An implementation is also available in the Flux package
# (https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse).
meansquarederror(ŷ, y) = sum((ŷ .- y) .^ 2) / size(y, 2)

function train(args, savename, savedir)

    # Check if model directory is created 
    !ispath(savedir) && mkpath(savedir)
    modelpath = joinpath(savedir, savename)

    # Load the data
    (x_train, y_train), (x_test, y_test) = get_processed_data(args)

    # The model
    m = model((randn(1, 13)), [0.0])

    loss(x, y) = meansquarederror(predict(model; x), y)

    ## Training
    η = args.lr
    θ = params(m.W, m.b)

    for i = 1:500
        g = gradient(() -> loss(x_train, y_train), θ)
        for x in θ
            update!(x, g[x] * η)
        end
        if i % 100 == 0
            @show loss(x_train, y_train)
        end
    end

    # Predict the RMSE on the test set
    err = meansquarederror(predict(m; x_test), y_test)
    println(format("Test error {}", err))
    @info format("Finished training")
    # Save the model and epoch and if training is done -- to resume 
    BSON.@save modelpath model
end

cd(@__DIR__)
# Initialize the Hyperparamters
args = Hyperparams()
# Train an ensemble of models 
ensemble_train(args, train)
# Evaluate ensemble
ensemble_evaluate(args, testloader, predict)
