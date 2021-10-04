#= 
 Train a simple network to predict formation energy per atom (downloaded from Materials Project). =#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs, glorot_uniform, Zygote, gpu
using ChemistryFeaturization
using AtomicGraphNets
using Formatting

include("models.jl")

function train_formation_energy(;
    num_pts = 100,
    num_epochs = 25,
    data_dir = joinpath(@__DIR__, "data"),
    verbose = true,
    sample_size = 10,
    complexity_constant = 1e-6,
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
    num_conv = 3 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?

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
        joinpath(data_dir, format("{}_cifs", prop), string(r[Symbol(id)], ".cif")) for
        r in eachrow(info)
    ]
    graphs = skipmissing(AtomGraph.(cifpaths))
    inputs = [featurize(gr, featurization) for gr in graphs]

    # pick out train/test sets
    if verbose
        println("Dividing into train/test sets...")
    end
    train_output = output[1:num_train]
    test_output = output[num_train+1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train+1:end]
    train_data = zip(train_input, train_output)
    test_data = zip(test_input, test_output)

    # build the model
    if verbose
        println("Building the network...")
    end
    model = BayesianCGCNN(
        num_features,
        num_conv = num_conv,
        atom_conv_feature_length = crys_fea_len,
        pooled_feature_length = (Int(crys_fea_len / 2)),
        num_hidden_layers = 1,
    )
    model = gpu(model)
    ps = Flux.params(model)
    opt = ADAM(0.01) # optimizer

    # define loss function and a callback to monitor progress
    loss(x, y) = Flux.Losses.mse(model(x), y)

    if verbose
        println("Training!")
    end

    function test(test_data)
        total_loss = 0
        count = 0
        for (x, y) in test_data
            x, y = x |> gpu, y |> gpu
            total_loss += loss(x, y)
            count += 1
        end
        return (total_loss / count)
    end

    function kl_loss_calc(model)
        layers = Zygote.@ignore Flux.modules(model)
        return sum(normal_kl_divergence.(layers))
    end
    for epoch = 1:num_epochs
        for (x, y) in train_data
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(ps) do
                for sample in sample_size
                    total_loss = loss(x, y)
                    kl_loss = kl_loss_calc(model)
                    total_loss += complexity_constant * kl_loss
                end
                total_loss /= sample_size
                return total_loss
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        test_loss = test(test_data)
        println(format("Test Loss: {}", test_loss))
    end

    return model
end

train_formation_energy()
