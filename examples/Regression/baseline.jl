#= 
 Train a simple network to predict formation energy per atom (downloaded from Materials Project). =#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs, glorot_uniform
using ChemistryFeaturization
using AtomicGraphNets
using Formatting
using Distributions
using CalibrationErrors
using CalibrationErrorsDistributions
using CalibrationTests

include("models.jl")
include("utils.jl")

function train_formation_energy(; num_epochs = 25, verbose = true)
    # model hyperparameters – keeping it pretty simple for now
    num_conv = 3 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
    num_features, xtrain, ytrain, xtest, ytest = get_data()
    train_data = zip(xtrain, ytrain)
    test_data = zip(xtest, ytest)

    model = CGCNN(
        num_features,
        num_conv = num_conv,
        atom_conv_feature_length = crys_fea_len,
        pooled_feature_length = (Int(crys_fea_len / 2)),
        num_hidden_layers = num_hidden_layers,
    )

    ps = Flux.params(model)
    opt = Nesterov(0.001) # optimizer

    # define loss function and a callback to monitor progress
    loss(x, y) = Flux.Losses.mse(model(x), y)
    # train
    if verbose
        println("Training!")
    end

    function get_preds(data)
        predictions = []
        targets = []
        for (x, y) in data
            preds = model(x)
            push!(predictions, preds[1])
            push!(targets, y)
        end
        return predictions, targets
    end

    # define kernel
    kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

    for epoch = 1:num_epochs
        train_loss = 0
        Flux.train!(loss, ps, train_data, opt)
        train_preds, train_targets = get_preds(train_data)
        test_preds, test_targets = get_preds(test_data)
        test_loss = Flux.Losses.mse(test_preds, test_targets) |> round4
        train_loss = Flux.Losses.mse(train_preds, train_targets) |> round4
        sigma = sqrt(train_loss)
        predictions = [Normal(mean, sigma) for mean in test_preds]

        # unbiased estimator of SKCE
        unbiased_estimator = UnbiasedSKCE(kernel)
        skce = calibrationerror(unbiased_estimator, predictions, test_targets) |> round4

        # biased estimator of SKCE
        biased_estimator = BiasedSKCE(kernel)
        biased_skce =
            calibrationerror(biased_estimator, predictions, test_targets) |> round4

        println(format("Train Loss: {}", train_loss))
        println(format("Test Loss: {}", test_loss))
        println(format("Unbiased SKCE: {}", skce))
        println(format("Biased SKCE: {}", biased_skce))
        println("===============")
    end

    return model
end

train_formation_energy()
