using Flux 
using BSON
using Formatting


"""
    ensemble_train(ensemble_obj, model, trainloder, train_function)

Trains an ensemble of N models defined in the ensemble_object. 
Each member of the ensemble (a copy of the given model) is trained 
with a different random seed and saved in the given directory. 

The train function is expected to have the logic for logging, evaluation 
and such as you would usually write to train a single model 
"""
function ensemble_train(args, train_function)
    for ensemble_model in 1:args.ensemble_size
        # Alter save path and model name to include model id 
        savepath = format("{}/{}_{}/",
                            args.savepath,
                            args.model_name, 
                            ensemble_model) 
        savename = format("{}_{}.BSON", args.model_name, ensemble_model)
        # Make sure the directory exists 
        !ispath(savepath) && mkpath(savepath)
        # Train the model 
        train_function(args, savename, savepath)
    end 
end

function ensemble_evaluate(args, testloader, prediction_function)
    predictions = [] 
    targets = nothing 
    # Individual model's metrics 
    model_metrics = [] 
    for ensemble_model in 1:args.ensemble_size
        # Alter save path and model name to include model id
        savepath = format("{}/{}_{}/",
                            args.savepath,
                            args.model_name, 
                            ensemble_model) 
        savename = format("{}_{}.BSON", args.model_name, ensemble_model)
        modelpath = format("{}{}", savepath, savename)
        # Try loading the model 
        model = nothing 
        try 
            BSON.@load modelpath model
            @info format("Loaded model {} successfully", ensemble_model)
        catch exception 
            @error format("Loading model {} failed: {}", ensemble_model, exception)
        end 

        # Get the predictions 
        model_predictions, model_targets = prediction_function(model, testloader)
        push!(predictions, model_predictions)
        targets = model_targets
    end 
    # Batch all model preductions 
    predictions = Flux.batch(predictions)
    println(size(targets))
    exit()
end 
    