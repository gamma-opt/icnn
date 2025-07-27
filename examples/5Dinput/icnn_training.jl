using Dates
using Gogeta
using Gurobi
using JuMP
using JSON

include("../../src/bound.jl")
include("../../src/branch.jl")
include("../../src/envelope.jl")
include("../../src/util.jl")
include("../../src/icnn.jl")

# Generate sample data
function generate_data(num_samples=1000)
    x = rand(Float32, 5, num_samples) .- 1
    y = sum(x.^2, dims=1)  # Convex function: sum of squares
    return x, y
end

# Create custom configuration
custom_config = ICNNConfig(
    5,                      # input_dim
    [20,10],                # hidden_dims
    1,                      # output_dim
    relu,                   # activation
    true,                   # use_skip_connections
    true,                   # use_convex_projection
    0.1f0,                  # learning_rate
    512,                    # batch_size
    200,                    # max_epochs
    50,                     # patience
    0.1f0,                  # dropout_rate - add some regularisation
    Flux.kaiming_normal,    # weight_init
    123                     # seed
)

# Generate data
x_train, y_train = generate_data(2000)  # More training data
x_val, y_val = generate_data(400)

println("Data shapes:")
println("  Training: ", size(x_train), " -> ", size(y_train))
println("  Validation: ", size(x_val), " -> ", size(y_val))

# Create and train model
model = ICNN(custom_config)
training_history = train_icnn(model, x_train, y_train, x_val, y_val)

# Save the weights to a JSON file
file_path = joinpath(@__DIR__, "icnn_model.json")
save_model(model, file_path)

# Print model summary
print_model_summary(model)

### base icnn model ###
icnn_lp =  Model(Gurobi.Optimizer)

icnn_output_var = @variable(icnn_lp, z, base_name="output_var")
icnn_input_var = @variable(icnn_lp, x[1:5], base_name="input_var")
@objective(icnn_lp, Min, 0)

ICNN_incorporate!(icnn_lp, file_path, icnn_output_var, icnn_input_var...)

### Branch and Bound setup ###
lb = [-1.0, -1.0, -1.0, -1.0, -1.0];
ub = [0.0, 0.0, 0.0, 0.0, 0.0];
box = Box(5, lb, ub)

tree_status = TreeStatus(
    -Inf,           # obj_lb: lower bound on the objective function
    nothing,        # x_optimal: optimal x values found so far
    [(box, 1)],     # bounds_to_branch: boxes to branch on, branching on dimension 1
    false           # all_pruned: flag to indicate if all branches are pruned
)

### level 0 root ###
root_icnn_lp = copy(icnn_lp);
set_optimizer(root_icnn_lp, Gurobi.Optimizer)

for i in 1:box.n
    @constraint(root_icnn_lp, root_icnn_lp[:x][i] >= box.lb[i])  # Lower bound constraint
    @constraint(root_icnn_lp, root_icnn_lp[:x][i] <= box.ub[i])  # Upper bound constraint
end

# add additional contraint on the output variable
@constraint(root_icnn_lp, root_icnn_lp[:z] >= 2.5)

# overwrite the objective
@objective(root_icnn_lp, Max, sum(root_icnn_lp[:x][i] for i in 1:5))

### B&B ###
branch_and_bound(icnn_lp, root_icnn_lp, tree_status)
