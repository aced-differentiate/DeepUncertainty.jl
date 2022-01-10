loss(ŷ, y) = logitcrossentropy(ŷ, y)

num_params(model) = sum(length, Flux.params(model))

round4(x) = round(x, digits=4)

function accuracy(preds, labels)
    acc = sum(onecold(preds |> cpu) .== onecold(labels |> cpu))
    return acc
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader(
        (xtrain, ytrain),
        batchsize=args.batchsize,
        shuffle=true,
        partial=false,
    )
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)

    # Fashion mnist for uncertainty testing 
    xtrain, ytrain = MLDatasets.FashionMNIST.traindata(Float32)
    xtest, ytest = MLDatasets.FashionMNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    ood_train_loader = DataLoader(
        (xtrain, ytrain),
        batchsize=args.batchsize,
        shuffle=true,
        partial=false,
    )
    ood_test_loader =
        DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)

    return train_loader, test_loader, ood_train_loader, ood_test_loader
end

function ensembles_evaluation(args, loader, model, device)
    l = [0.0f0 for x = 1:args.ensemble_size]
    acc = [0 for x = 1:args.ensemble_size]
    ece_list = [0.0f0 for x = 1:args.ensemble_size]
    ntot = 0
    mean_l = 0
    mean_acc = 0
    mean_ece = 0
    mean_entropy = 0 
    for (x, y) in loader
        x = repeat(x, 1, 1, 1, args.ensemble_size)
        x, y = x |> device, y |> device
        # Perform the forward pass 
        ŷ = model(x)
        # Reshape the predictions into [classes, batch_size, ensemble_size
        reshaped_ŷ = reshape(ŷ, size(ŷ)[1], args.batchsize, args.ensemble_size)
        # Loop through each model's predictions 
        for ensemble = 1:args.ensemble_size
            model_predictions = reshaped_ŷ[:, :, ensemble]
            # Calculate individual loss 
            l[ensemble] += loss(model_predictions, y) * size(model_predictions)[end]
            acc[ensemble] += accuracy(model_predictions, y)
            ece_list[ensemble] +=
                expected_calibration_error(model_predictions, onecold(y)) * args.batchsize
        end
        # Get the mean predictions
        mean_predictions = mean(reshaped_ŷ, dims=ndims(reshaped_ŷ))
        mean_predictions = dropdims(mean_predictions, dims=ndims(mean_predictions))
        mean_l += loss(mean_predictions, y) * size(mean_predictions)[end]
        mean_acc += accuracy(mean_predictions, y)
        mean_ece +=
            expected_calibration_error(mean_predictions, onecold(y)) * args.batchsize
        mean_entropy += mean(calculate_entropy(mean_predictions)) * args.batchsize
        ntot += size(mean_predictions)[end]
    end
    # Normalize the loss 
    losses = round4.(l ./ ntot)
    acc = round4.((acc ./ ntot) .* 100)
    ece_list = round4.(ece_list ./ ntot)
    # Calculate mean loss 
    mean_l = round4(mean_l / ntot)
    mean_acc = round4((mean_acc / ntot) .* 100)
    mean_ece = round4(mean_ece / ntot)
    mean_entropy = mean_entropy / ntot |> round4

    # Print the per ensemble mode loss and accuracy 
    for ensemble = 1:args.ensemble_size
        @info (format(
            "Model {} Loss: {} Accuracy: {} ECE: {}",
            ensemble,
            losses[ensemble],
            acc[ensemble],
            ece_list[ensemble],
        ))
    end
    @info (format(
        "Mean Loss: {} Mean Accuracy: {} Mean ECE: {} Mean Entropy: {}",
        mean_l,
        mean_acc,
        mean_ece,
        mean_entropy, 
    ))
    @info "========================="
    return nothing
end

function monte_carlo_evaluation(args, loader, model, device)
    ntot = 0
    mean_l = 0
    mean_acc = 0
    mean_ece = 0
    mean_entropy = 0
    for (x, y) in loader
        predictions = []
        x, y = x |> device, y |> device

        # Loop through each model's predictions 
        for ensemble = 1:args.sample_size
            model_predictions = model(x)
            push!(predictions, model_predictions)
        end
        # Get the mean predictions
        predictions = Flux.batch(predictions)
        mean_predictions = mean(predictions, dims=ndims(predictions))
        mean_predictions = dropdims(mean_predictions, dims=ndims(mean_predictions))
        mean_l += loss(mean_predictions, y) * size(mean_predictions)[end]
        mean_acc += accuracy(mean_predictions, y)
        mean_ece +=
            expected_calibration_error(mean_predictions, onecold(y)) * args.batchsize
        mean_entropy += mean(calculate_entropy(mean_predictions)) * args.batchsize
        ntot += size(mean_predictions)[end]
    end
    # Calculate mean loss 
    mean_l = mean_l / ntot |> round4
    mean_acc = mean_acc / ntot * 100 |> round4
    mean_ece = mean_ece / ntot |> round4
    mean_entropy = mean_entropy / ntot |> round4

    # Print the per ensemble mode loss and accuracy 
    @info (format(
        "Mean Loss: {} Mean Accuracy: {} Mean ECE: {} Mean Entropy: {}",
        mean_l,
        mean_acc,
        mean_ece,
        mean_entropy,
    ))
    @info "========================="
    return nothing
end
