using Flux
using Random
using Statistics
using LinearAlgebra
using JSON

# Configuration structure for ICNN
struct ICNNConfig
    input_dim::Int
    hidden_dims::Vector{Int}
    output_dim::Int
    activation::Function
    use_skip_connections::Bool
    use_convex_projection::Bool
    learning_rate::Float32
    batch_size::Int
    max_epochs::Int
    patience::Int
    dropout_rate::Float32
    weight_init::Function
    seed::Union{Int, Nothing}
end

# Flexible ICNN Model
struct ICNN
    config::ICNNConfig
    layers::Vector{Dense}
    skip_layers::Vector{Union{Dense, Nothing}}
    dropout_layers::Vector{Union{Flux.Dropout, Nothing}}
end

# Constructor for ICNN
function ICNN(config::ICNNConfig)
    # Set random seed if provided
    if config.seed !== nothing
        Random.seed!(config.seed)
    end
    
    activation = config.activation
    init_fn = config.weight_init
    
    layers = Dense[]
    skip_layers = Union{Dense, Nothing}[]
    dropout_layers = Union{Flux.Dropout, Nothing}[]
    
    # Input layer
    first_hidden = config.hidden_dims[1]
    push!(layers, Dense(config.input_dim => first_hidden, activation; init=init_fn))
    push!(skip_layers, nothing)  # No skip for input layer
    
    # Add dropout if specified
    if config.dropout_rate > 0
        push!(dropout_layers, Flux.Dropout(config.dropout_rate))
    else
        push!(dropout_layers, nothing)
    end
    
    # Hidden layers
    for i in 2:length(config.hidden_dims)
        prev_dim = config.hidden_dims[i-1]
        curr_dim = config.hidden_dims[i]
        
        # Main layer (weights will be projected to non-negative)
        # Activation is applied manually in the forward pass function
        push!(layers, Dense(prev_dim => curr_dim; init=init_fn))
        
        # Skip connection from input
        if config.use_skip_connections
            push!(skip_layers, Dense(config.input_dim => curr_dim, bias=false; init=init_fn))
        else
            push!(skip_layers, nothing)
        end
        
        # Dropout
        if config.dropout_rate > 0
            push!(dropout_layers, Flux.Dropout(config.dropout_rate))
        else
            push!(dropout_layers, nothing)
        end
    end
    
    # Output layer
    last_hidden = config.hidden_dims[end]
    push!(layers, Dense(last_hidden => config.output_dim; init=init_fn))
    
    # Final skip connection
    if config.use_skip_connections
        push!(skip_layers, Dense(config.input_dim => config.output_dim, bias=false; init=init_fn))
    else
        push!(skip_layers, nothing)
    end
    
    push!(dropout_layers, nothing)  # No dropout on output
    
    return ICNN(config, layers, skip_layers, dropout_layers)
end

# Forward pass
function (model::ICNN)(x; training=false)
    activation = model.config.activation
    current = x
    
    # First layer (input -> first hidden)
    current = model.layers[1](current)
    if model.dropout_layers[1] !== nothing && training
        current = model.dropout_layers[1](current)
    end
    
    # Hidden layers with skip connections
    for i in 2:(length(model.layers)-1)
        # Main path
        next_layer = model.layers[i](current)
        
        # Add skip connection if exists
        if model.skip_layers[i] !== nothing
            skip_connection = model.skip_layers[i](x)
            next_layer = next_layer + skip_connection
        end
        
        # Apply activation
        current = activation.(next_layer)
        
        # Apply dropout if exists and training
        if model.dropout_layers[i] !== nothing && training
            current = model.dropout_layers[i](current)
        end
    end
    
    # Output layer
    output = model.layers[end](current)
    
    # Final skip connection
    if model.skip_layers[end] !== nothing
        output = output + model.skip_layers[end](x)
    end
    
    return output
end

# Project weights to maintain convexity (non-negative weights for hidden layers)
function project_convex_weights!(model::ICNN)
    # Project all hidden layer weights (not first layer, not skip connections)
    for i in 2:length(model.layers)
        model.layers[i].weight .= max.(model.layers[i].weight, 0.0f0)
    end
end

# Training function
function train_icnn(model::ICNN, x_train, y_train, x_val, y_val; verbose=true)
    config = model.config
    
    # Loss function
    loss_fn(x, y) = Flux.mse(model(x; training=true), y)
    
    # Optimiser
    optimiser = Flux.Adam(config.learning_rate)
    params = Flux.params(model)
    
    # Early stopping variables
    best_val_loss = Inf32
    epochs_without_improvement = 0
    best_model_state = deepcopy(model)
    
    # Training metrics
    train_losses = Float32[]
    val_losses = Float32[]
    
    steps_per_epoch = size(x_train, 2) ÷ config.batch_size
    
    if verbose
        println("Starting ICNN training...")
        println("Configuration:")
        println("  Architecture: $(config.input_dim) -> $(join(config.hidden_dims, " -> ")) -> $(config.output_dim)")
        println("  Activation: $(config.activation)")
        println("  Skip connections: $(config.use_skip_connections)")
        println("  Convex projection: $(config.use_convex_projection)")
        println("  Learning rate: $(config.learning_rate)")
        println("  Batch size: $(config.batch_size)")
        println("  Max epochs: $(config.max_epochs)")
        println("  Patience: $(config.patience)")
        println("  Dropout: $(config.dropout_rate)")
        println("  Steps per epoch: $steps_per_epoch")
        println()
    end
    
    for epoch in 1:config.max_epochs
        epoch_losses = Float32[]
        
        # Training loop
        for step in 1:steps_per_epoch
            start_idx = (step - 1) * config.batch_size + 1
            end_idx = step * config.batch_size
            
            x_batch = x_train[:, start_idx:end_idx]
            y_batch = y_train[:, start_idx:end_idx]
            
            # Compute gradients and update
            loss_val = 0.0f0
            grads = Flux.gradient(params) do
                loss_val = loss_fn(x_batch, y_batch)
                return loss_val
            end
            
            Flux.update!(optimiser, params, grads)
            if config.use_convex_projection
                project_convex_weights!(model)
            end
            
            push!(epoch_losses, loss_val)
        end
        
        # Validation
        val_loss = Flux.mse(model(x_val; training=false), y_val)
        avg_train_loss = mean(epoch_losses)
        
        push!(train_losses, avg_train_loss)
        push!(val_losses, val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = deepcopy(model)
            if verbose
                println("Epoch $epoch, Train: $(round(avg_train_loss, digits=6)), Val: $(round(val_loss, digits=6)) (best)")
            end
        else
            epochs_without_improvement += 1
            if verbose
                println("Epoch $epoch, Train: $(round(avg_train_loss, digits=6)), Val: $(round(val_loss, digits=6)) ($(epochs_without_improvement)/$(config.patience))")
            end
            
            if epochs_without_improvement >= config.patience
                if verbose
                    println("\nEarly stopping! Best val loss: $(round(best_val_loss, digits=6))")
                end
                break
            end
        end
    end
    
    # Restore best model
    for i in 1:length(model.layers)
        model.layers[i] = best_model_state.layers[i]
    end
    for i in 1:length(model.skip_layers)
        model.skip_layers[i] = best_model_state.skip_layers[i]
    end
    
    return Dict("train_losses" => train_losses, "val_losses" => val_losses, "best_val_loss" => best_val_loss)
end

# Extract weights in Gogeta-style with named layers
# Weights: W[layer name][1][column num][row]
# Biases: W[layer name][2][bias index]
# names SKIP-2,3,...,k (skip connection)
#       FC-1,2,...,k (fully connected)
#       (k is output layer index)
function save_model(model::ICNN, json_file_path::String)
    weights_json = Dict{String, Any}()

    for (i, layer) in enumerate(model.layers)
        # Determine the name of the main layer
        layer_name = "FC$i"
        weights_list = [layer.weight[:, r] for r in axes(layer.weight, 2)]
        biases_list = layer.bias
        weights_json[layer_name] = Any[weights_list, biases_list]
    end

    for (i, skip_layer) in enumerate(model.skip_layers)
        if skip_layer !== nothing
            skip_name = "SKIP$i"
            weights_list = [skip_layer.weight[:, r] for r in axes(skip_layer.weight, 2)]
            weights_json[skip_name] = Any[weights_list]
        end
    end

    # Save the weights to a JSON file
    open(json_file_path, "w") do io
        write(io, JSON.json(weights_json, 4))
    end
end

# Utility function to print layer information (like Python model.summary())
function print_model_summary(model::ICNN)
    println("ICNN Model Summary:")
    println("=" ^ 50)
    
    total_params = 0
    
    # Main layers
    for (i, layer) in enumerate(model.layers)
        layer_name = "FC$i"
        weight_shape = size(layer.weight)
        bias_params = layer.bias == false ? 0 : length(layer.bias)
        layer_params = prod(weight_shape) + bias_params
        total_params += layer_params
        
        println("$layer_name: $(weight_shape[2]) -> $(weight_shape[1]) ($(layer_params) params)")
    end
    
    # Skip layers
    for (i, skip_layer) in enumerate(model.skip_layers)
        if skip_layer !== nothing
            layer_name = "SKIP$i"
            weight_shape = size(skip_layer.weight)
            bias_params = skip_layer.bias == false ? 0 : length(skip_layer.bias)
            layer_params = prod(weight_shape) + bias_params
            total_params += layer_params
            
            println("$layer_name: $(weight_shape[2]) -> $(weight_shape[1]) ($(layer_params) params)")
        end
    end
    
    println("=" ^ 50)
    println("Total parameters: $total_params")
    println("Configuration: $(model.config.activation) activation, skip_connections=$(model.config.use_skip_connections), convex_projection=$(model.config.use_convex_projection)")
end
