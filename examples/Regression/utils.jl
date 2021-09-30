using CSV, DataFrames
using Random, Statistics
using ChemistryFeaturization
using AtomicGraphNets
using Formatting
using CalibrationErrors

round4(x) = round(x, digits = 4)

function get_data(num_pts = 100, data_dir = joinpath(@__DIR__, "data"), verbose = true)
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
    # model hyperparameters – keeping it pretty simple for now
    num_conv = 3 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?

    model = CGCNN(
        num_features,
        num_conv = num_conv,
        atom_conv_feature_length = crys_fea_len,
        pooled_feature_length = (Int(crys_fea_len / 2)),
        num_hidden_layers = num_hidden_layers,
    )

    return model, train_data, test_data
end
