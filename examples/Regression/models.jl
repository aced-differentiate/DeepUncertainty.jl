using DeepUncertainty
include("layers.jl")

function CGCNN(
    input_feature_length;
    num_conv = 2,
    conv_activation = softplus,
    atom_conv_feature_length = 80,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    num_hidden_layers = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    model = Chain(
        AGNConv(
            input_feature_length => atom_conv_feature_length,
            conv_activation,
            initW = initW,
        ),
        [
            AGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
        [
            Dense(
                pooled_feature_length,
                pooled_feature_length,
                hidden_layer_activation,
                init = initW,
            ) for i = 1:num_hidden_layers-1
        ]...,
        Dense(pooled_feature_length, output_length, output_layer_activation, init = initW),
    )
    return model
end

function MC_CGCNN(
    input_feature_length;
    num_conv = 2,
    conv_activation = softplus,
    atom_conv_feature_length = 80,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    num_hidden_layers = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
    dropout_rate = 0.2,
)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    model = Chain(
        MCAGNConv(
            input_feature_length => atom_conv_feature_length,
            dropout_rate,
            conv_activation,
            initW = initW,
        ),
        [
            MCAGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                dropout_rate,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
        [
            MCLayer(
                MCDense(
                    pooled_feature_length,
                    pooled_feature_length,
                    dropout_rate,
                    hidden_layer_activation,
                ),
                dropout_rate,
            ) for i = 1:num_hidden_layers-1
        ]...,
        MCDense(
            pooled_feature_length,
            output_length,
            dropout_rate,
            output_layer_activation,
        ),
    )
    return model
end

function BayesianCGCNN(
    input_feature_length;
    num_conv = 2,
    conv_activation = softplus,
    atom_conv_feature_length = 80,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    num_hidden_layers = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    model = Chain(
        VariationalAGNConv(
            input_feature_length => atom_conv_feature_length,
            conv_activation,
            initW = initW,
        ),
        [
            VariationalAGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
        [
            VariationalDense(
                pooled_feature_length,
                pooled_feature_length,
                hidden_layer_activation,
                init = initW,
            ) for i = 1:num_hidden_layers-1
        ]...,
        VariationalDense(
            pooled_feature_length,
            output_length,
            output_layer_activation,
            init = initW,
        ),
    )
    return model
end
