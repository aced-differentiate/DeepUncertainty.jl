using Flux 
using BSON
using Formatting

"""
    Ensemble(args, save_dir, ensemble_size = 10)

Train and evaluate an ensemble of models given the dataset 
and the training/testing loops. 

# Parameters 
- Experiment argumens ('args'): All the command line arguments as a dict to be 
                                stored and used during training/testing.
- Save directory ('save_dir'):  The directory where logs and trained models are 
                                stored. The trained models are stored in the 
                                following fashion -- save_dir/run_no/tb_events, model.bson
- Ensemble size ('ensemble_size'):  The total number of models in the ensemble 
"""
mutable struct Ensemble 
    args 
    save_dir::String 
    ensemble_size::Int
end 

"""
    ensemble_train(ensemble_obj, model, trainloder, train_function)

Trains an ensemble of N models defined in the ensemble_object. 
Each member of the ensemble (a copy of the given model) is trained 
with a different random seed and saved in the given directory. 

The train function is expected to have the logic for logging, evaluation 
and such as you would usually write to train a single model 
"""
function ensemble_train(obj::Ensemble, train_function)
    for ensemble_model in 1:obj.ensemble_size
        model = train_function() 

        model_name = format("model_{}.bson", ensemble_model)
        model_path = format("{}/{}/model_{}",
                            obj.args.savepath,
                            obj.args.model_name, 
                            ensemble_model) 

        # Check if model directory is created 
        !ispath(model_path) && mkpath(model_path)
        modelpath = joinpath(model_path, model_name) 
        BSON.@save modelpath model
        @info "Model \"$(ensemble_model)\" saved in \"$(modelpath)\""
    end 
end