#= 
 Train a simple network to predict formation energy per atom (downloaded from Materials Project). =#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux:@epochs
using ChemistryFeaturization
using AtomicGraphNets
using Formatting
include("models.jl")
using Distributions
using CalibrationErrors
using CalibrationErrorsDistributions
using CalibrationTests

function train_formation_energy(;
    num_pts=5000,
    num_epochs=25,
    data_dir=joinpath(@__DIR__, "data"),
    verbose=true,
)
    println("Setting things up...")

    # data-related options
    train_frac = 0.8 # what fraction for training?
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    prop = "formation_energy_per_atom"
    id = "task_id" # field by which to label each input material

    # set up the featurization
    featurization = GraphNodeFeaturization([
        "Group",
        "Row",
        "Block",
        "Atomic mass",
        "Atomic radius",
        "X",
    ])
    num_features =
        sum(ChemistryFeaturization.FeatureDescriptor.output_shape.(featurization.features)) # TODO: update this with cleaner syntax once new version of CF is tagged that has it

    # model hyperparameters – keeping it pretty simple for now
    num_conv = 4 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
    opt = ADAM(0.001) # optimizer

    # dataset...first, read in outputs
    info = CSV.read(string(data_dir, "/", prop, ".csv"), DataFrame)
    y = Array(Float32.(info[!, Symbol(prop)]))

    # shuffle data and pick out subset
    indices = shuffle(1:size(info, 1))[1:num_pts]
    info = info[indices, :]
    output = y[indices]

    # next, make and featurize graphs
    if verbose
        println("Building graphs and feature vectors from structures...")
    end
    inputs = FeaturizedAtoms[]

    # for r in eachrow(info)
    cifpaths = [
        joinpath(data_dir, format("{}_cifs", prop), string(r[Symbol(id)], ".cif")) for r in eachrow(info)
    ]

    outputs = []
    for (cifpath, label) in zip(cifpaths, output)
        try 
            graph = AtomGraph.(cifpath)
            input = featurize(graph, featurization)
            push!(inputs, input)
            push!(outputs, label)
        catch
            continue 
        end 
    end 

    # pick out train/test sets
    if verbose
        println("Dividing into train/test sets...")
    end
    train_output = outputs[1:num_train]
    test_output = outputs[num_train + 1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train + 1:end]
    train_data = zip(train_input, train_output)
    test_data = zip(test_input, test_output)

    # build the model
    if verbose
        println("Building the network...")
    end
    model = build_CGCNN(
        num_features,
        num_conv=num_conv,
        atom_conv_feature_length=crys_fea_len,
        pooled_feature_length=(Int(crys_fea_len / 2)),
        num_hidden_layers=num_hidden_layers,
    )

    # define loss function and a callback to monitor progress
    loss(x, y) = Flux.Losses.mse(model(x), y)
    evalcb_verbose() = @show(mean(loss.(test_input, test_output)))
    evalcb_quiet() = return nothing
    evalcb = verbose ? evalcb_verbose : evalcb_quiet
    evalcb()

    # train
    if verbose
        println("Training!")
    end
    # @epochs num_epochs Flux.train!(
    #     loss,
    #     Flux.params(model),
    #     train_data,
    #     opt,
    #     cb=Flux.throttle(evalcb, 5),
    # )
    num_samples = 1
    function get_preds(data)
        predictions = []
        targets = []
        samples = []
        for (x, y) in data
            preds = model(x)    
            push!(predictions, preds[1])
            push!(targets, y)
        end
        return predictions, targets
    end

    ps = Flux.params(model)
    # define kernel
    kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

    for epoch = 1:num_epochs
        train_loss = 0
        n_tot = 0 
        # Flux.train!(loss, ps, train_data, opt)
        for (x, y) in train_data
            gs = Flux.gradient(ps) do
                total_loss = loss(x, y)
                train_loss += total_loss
                return total_loss
            end 
            Flux.Optimise.update!(opt, ps, gs)
            n_tot += 1 
        end
        train_loss = train_loss / n_tot
        @info(format("Epoch {}", epoch))
        @info(format("Train Loss: {}", train_loss))

        test_preds, test_targets = get_preds(test_data) 
        test_loss = mean(Flux.Losses.mse.(test_preds, test_targets))

        sigma = sqrt(train_loss)
        predictions = [Normal(mean, sigma) for mean in test_preds]
        # unbiased estimator of SKCE
        unbiased_estimator = UnbiasedSKCE(kernel)
        skce = calibrationerror(unbiased_estimator, predictions, test_targets)
        # biased estimator of SKCE
        biased_estimator = BiasedSKCE(kernel)
        biased_skce =
            calibrationerror(biased_estimator, predictions, test_targets)

        @info(format("Epoch {}", epoch))
        @info(format("Train Loss: {}", train_loss))
        @info(format("Test Loss: {}", test_loss))
        @info(format("Unbiased SKCE: {}", skce))
        @info(format("Biased SKCE: {}", biased_skce))
        @info("===============")
    end 
    return model
end

train_formation_energy()