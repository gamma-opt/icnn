using Gogeta
using Plots
using JSON

include("../../src/bound.jl")
include("../../src/branch.jl")
include("../../src/envelope.jl")
include("../../src/solver.jl")
include("../../src/util.jl")

# replace with the path to the model_weights.json file
file_path = joinpath(@__DIR__, "model_weights_3D.json")

#-------------------------------------ICNN-----------------------------------------#

# define the base icnn model without any additional constraints
icnn_lp =  Model(alpine_optimiser)

# define the output variable
icnn_output_var = @variable(icnn_lp, z, base_name="output_var")

# define the input variable as a vector with 2 elements
icnn_input_var = @variable(icnn_lp, x[1:2], base_name="input_var")

# define the objective function
@objective(icnn_lp, Min, 0)

# reformulate the model
ICNN_incorporate!(icnn_lp, file_path, icnn_output_var, icnn_input_var...)

print(icnn_lp)  # min output_var z

#-----------------------------Branch and Bound setup--------------------------------#

lb = [-1.0, -1.0]
ub = [1.0, 1.0]
box = Box(2, lb, ub)

tree_status = TreeStatus(
    -Inf,           # obj_lb: lower bound on the objective function
    nothing,        # x_optimal: optimal x values found so far
    [(box, 1)],     # bounds_to_branch: boxes to branch on, branching on dimension 1
    false           # all_pruned: flag to indicate if all branches are pruned
)

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
tree_status = process_results(results, tree_status)

#---------------Option 1: then start branching using B&B manually-----------------#
#-----------------------------------level 1---------------------------------------#

# Branch the box on the first dimension (x[1]) to create two new boxes
tree_status.bounds_to_branch= branch_box(tree_status.bounds_to_branch)

new_icnn_lp_list = generate_relaxation(icnn_lp, root_icnn_lp, tree_status)
results = solve_node_models(new_icnn_lp_list, icnn_lp)

tree_status = process_results(results, tree_status)

#------------Option 2: then start branching using B&B up-level function-----------#
x_optimal, obj_optimal = branch_and_bound(icnn_lp, root_icnn_lp, tree_status)
