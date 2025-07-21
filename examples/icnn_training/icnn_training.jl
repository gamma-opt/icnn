using JuMP
using Gurobi
using Gogeta

include("../../src/icnn.jl")

# Generate sample data
function generate_data(num_samples=1000)
    x = rand(Float32, 2, num_samples) * 2 .- 1
    y = sum(x.^2, dims=1)  # Convex function: sum of squares
    return x, y
end

# Example usage with custom configuration
println("=== Custom ICNN Example ===")

# Create custom configuration
custom_config = ICNNConfig(
    2,                      # input_dim
    [64, 64],               # hidden_dims - 2 hidden layers
    1,                      # output_dim
    "relu",                 # activation
    true,                   # use_skip_connections
    0.001f0,               # learning_rate
    64,                     # batch_size - larger batch
    150,                    # max_epochs - more epochs
    15,                     # patience - more patience
    0.1f0,                 # dropout_rate - add some regularization
    "kaiming_uniform",      # weight_init
    123                     # seed
)

# Generate data
x_train, y_train = generate_data(2000)  # More training data
x_val, y_val = generate_data(400)

println("Data shapes:")
println("  Training: ", size(x_train), " -> ", size(y_train))
println("  Validation: ", size(x_val), " -> ", size(y_val))
println()

# Create and train model
model = ICNN(custom_config)
training_history = train_icnn(model, x_train, y_train, x_val, y_val)

# Test predictions
println("\n=== Testing ===")
x_test = Float32[0.5 -0.3 0.8; -0.5 0.7 -0.2]  # 3 test samples
y_pred = model(x_test; training=false)
y_expected = sum(x_test.^2, dims=1)

for i in 1:size(x_test, 2)
    pred = y_pred[1, i]
    expected = y_expected[1, i]
    error = abs(pred - expected)
    println("Sample $i: [$(round(x_test[1,i], digits=2)), $(round(x_test[2,i], digits=2))] -> Pred: $(round(pred, digits=4)), True: $(round(expected, digits=4)), Error: $(round(error, digits=4))")
end

# Save the weights to a JSON file
json_file_path = joinpath(@__DIR__, "icnn_model.json")
save_model(model, json_file_path)

# Print model summary
print_model_summary(model)

# Formulation of Input Convex Neural Networks (ICNNs) 
jump_model = Model(Gurobi.Optimizer);
set_silent(jump_model)

@variable(jump_model, -1 <= x <= 1);
@variable(jump_model, -1 <= y <= 1);

#variable associated with ICNNs output
@variable(jump_model, z);

@constraint(jump_model, y >= 1-x);
@objective(jump_model, Min, x+y);

ICNN_incorporate!(jump_model, json_file_path, z, x, y);

# see optimal solution
optimize!(jump_model)
x_opt = value(x)
y_opt = value(y)

println("Solution:")
println("x = ", x_opt)
println("y = " , y_opt)