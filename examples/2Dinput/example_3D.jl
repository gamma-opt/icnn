using Gogeta
using JuMP
using Plots
using JSON

include("../../src/bound.jl")
include("../../src/branch.jl")
include("../../src/envelope.jl")
include("../../src/solver.jl")
include("../../src/util.jl")

# replace with the path to the model_weights.json file
file_path = joinpath(@__DIR__, "model_weights_3D.json")

# Read the JSON file
json_content = read(file_path, String)

# Parse the JSON content
parsed_json = JSON.parse(json_content)

#-------------------------------------ICNN-----------------------------------------#

# define the base icnn model without any additional constraints, only box constraints on the input variable
icnn_lp =  Model(alpine_optimiser)

# define the output variable
icnn_output_var = @variable(icnn_lp, z, base_name="output_var")

# define the input variable as a vector with 2 elements
icnn_input_var = @variable(icnn_lp, x[1:2], base_name="input_var")

# # define the objective function
@objective(icnn_lp, Min, 0)

# add variables, constraints, and an objective function to the icnn_lp model to account for the generated ICNN.
ICNN_incorporate!(icnn_lp, file_path, icnn_output_var, icnn_input_var...)

print(icnn_lp)  # min output_var z

#-----------------------------Branch and Bound setup--------------------------------#

obj_lb = -Inf # updated upwards by feasible solutions (z_optimal gap within tolerance)
x_optimal = nothing # updated with the x values of the best solution found so far

lb = [-1.0, -1.0]
ub = [1.0, 1.0]
box = Box(2, lb, ub)

#-----------------------------level 0 root------------------------------------------#

# without BB method, directly optimise the ICNN model with additional constraints and custom objective (no penalty term)
# max x[1]+z s.t. z >= 0.5
root_icnn_lp = copy(icnn_lp)
set_optimizer(root_icnn_lp, alpine_optimiser)

# define the constraints on input variable
for i in 1:box.n
    @constraint(root_icnn_lp, root_icnn_lp[:x][i] >= box.lb[i])  # Lower bound constraint
    @constraint(root_icnn_lp, root_icnn_lp[:x][i] <= box.ub[i])  # Upper bound constraint
end

# add additional contraint on the output variable
@constraint(root_icnn_lp, root_icnn_lp[:z] >= 0.5)

# overwrite the objective
@objective(root_icnn_lp, Max, root_icnn_lp[:x][1] + root_icnn_lp[:z])

print(root_icnn_lp)

results = solve_node_models([root_icnn_lp], icnn_lp)
obj_lb, x_optimal, boxes_to_branch, all_pruned = process_results(results, [box], obj_lb, x_optimal)

#-------------------------then start branching using B&B--------------------------#
#-----------------------------------level 1---------------------------------------#

# Branch the box on the first dimension (x[1]) to create two new boxes
boxes_list = branch_box(box, branch_dimension=1)

new_icnn_lp_list = generate_relaxation(icnn_lp, root_icnn_lp, boxes_list)
results = solve_node_models(new_icnn_lp_list, icnn_lp)

obj_lb, x_optimal, boxes_to_branch, all_pruned = process_results(results, boxes_list, obj_lb, x_optimal)
