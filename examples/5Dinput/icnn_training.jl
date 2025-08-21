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
    0.01f0,                 # learning_rate
    32,                     # batch_size
    300,                    # max_epochs
    50,                     # patience
    0,                      # dropout_rate - add some regularisation
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

### NN->MIP ###
function set_solver!(jump)
    set_optimizer(jump, () -> Gurobi.Optimizer())
    set_silent(jump)
end

custom_config_nn = ICNNConfig(
    5,                      # input_dim
    [20,10],                # hidden_dims
    1,                      # output_dim
    relu,                   # activation
    false,                  # use_skip_connections
    false,                  # use_convex_projection
    0.01f0,                 # learning_rate
    64,                     # batch_size
    200,                    # max_epochs
    50,                     # patience
    0.1f0,                  # dropout_rate - add some regularisation
    Flux.glorot_uniform,    # weight_init
    123                     # seed
)

model_nn = ICNN(custom_config_nn)
training_history_nn = train_icnn(model_nn, x_train, y_train, x_val, y_val)

file_path_nn = joinpath(@__DIR__, "icnn_model_nn.json")
save_model(model_nn, file_path_nn)

lb = [-1.0, -1.0, -1.0, -1.0, -1.0];
ub = [0.0, 0.0, 0.0, 0.0, 0.0];

nn_mip =  Model(Gurobi.Optimizer)

nn_output_var = @variable(nn_mip, z, base_name="output_var")
nn_input_var = @variable(nn_mip, x[1:5], base_name="input_var")
@objective(nn_mip, Max, sum(nn_input_var[i] for i in 1:5))
@constraint(nn_mip, nn_output_var >= 2.5)
# @objective(nn_mip, Max, nn_output_var)

start_time = time()
NN_incorporate!(nn_mip, file_path_nn, nn_output_var, nn_input_var...; U_in=ub, L_in=lb, )
println("\nFormulated in $(round(time() - start_time, digits=4)) seconds")

start_time = time()
optimize!(nn_mip)
println("\nSolved in $(round(time() - start_time, digits=4)) seconds")
println("         Status: ", termination_status(nn_mip))
println("Objective value: ", objective_value(nn_mip))
println("        x value: ", value.(nn_mip[:x]))
println("        z value: ", value(nn_mip[:z]))
x = [0.0, -0.8076734819182435, -1.0, 0.0, -1.0]
y = sum(x.^2)