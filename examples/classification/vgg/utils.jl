using MLDatasets: CIFAR10, SVHN2

loss(ŷ, y) = logitcrossentropy(ŷ, y)

num_params(model) = sum(length, Flux.params(model))

round4(x) = round(x, digits = 4)

function get_data(args)
    x, y = CIFAR10.traindata()
    train_x = float(x)
    train_y = onehotbatch(y, 0:9)
    train_data = (train_x, train_y)

    test_x, test_y = CIFAR10.testdata()
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    test_data = (test_x, test_y)

    # OOD dataset
    ood_test_x, ood_test_y = SVHN2.testdata()
    ood_test_x = float(ood_test_x)
    ood_test_y = onehotbatch(ood_test_y, 1:10)
    ood_test_data = (ood_test_x, ood_test_y)

    train_loader =
        DataLoader(train_data, batchsize = args.batchsize, shuffle = true, partial = false)
    test_loader = DataLoader(test_data, batchsize = args.batchsize)
    ood_test_loader = DataLoader(ood_test_data, batchsize = args.batchsize)

    return train_loader, test_loader, ood_test_loader
end
