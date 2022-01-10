using CSV, DataFrames
using Random, Statistics
using ChemistryFeaturization
using AtomicGraphNets
using Formatting
using CalibrationErrors
using BSON: @save, @load
using BSON

round4(x) = round(x, digits=4)

function get_data(num_pts=100, data_dir=joinpath(@__DIR__, "data"), verbose=true)
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
        joinpath(data_dir, format("{}_cifs", prop), string(r[Symbol(id)], ".cif")) for r in eachrow(info)
    ]

    outputs = []
    inputs = []
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
    
    println(length(inputs), length(outputs))

    # graphs = skipmissing(AtomGraph.(cifpaths))
    # inputs = [featurize(gr, featurization) for gr in graphs

    # pick out train/test sets
    if verbose
        println("Dividing into train/test sets...")
    end 

    train_output = output[1:num_train]
    test_output = output[num_train + 1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train + 1:end]

    # build the model
    if verbose
        println("Building the network...")
    end

    return num_features, train_input, train_output, test_input, test_output
end
