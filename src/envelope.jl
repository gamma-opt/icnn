# define a function that generates a copy of icnn_lp and adds new constraints
"""
    add_envelope_constraints(filepath, icnn_lp, bounds_matrix, obj_sense=:Min, obj_expr=nothing)

This function takes an ICNN optimisation model and augments it with additional constraints to form an envelope around the ICNN. The envelope is defined by the convex hull of extreme points derived from the bounds of the input variables. The function generates a new optimisation model with the following features:

# Arguments:
- `filepath`: The path to the file containing the ICNN model definition.
- `icnn_lp`: The original ICNN optimisation model.
- `bounds_matrix`: A 2xN matrix where each column defines the lower and upper bounds for the corresponding input variable.
- `obj_sense`: The sense of the objective function (`:Min` or `:Max`). Defaults to `:Min`.
- `obj_expr`: An optional custom objective function that takes the input and output variables as arguments.

# Returns:
- The augmented optimisation model (`new_icnn_lp`) with the added envelope constraints.
"""

function add_envelope_constraints(filepath, icnn_lp, bounds_matrix, obj_sense=:Min, obj_expr=nothing)

    n = size(bounds_matrix, 2)  # Number of dimensions
    extreme_points = generate_extreme_combinations(bounds_matrix)
    num_extremes = size(extreme_points, 2)

    # define the icnn model
    new_icnn_lp =  Model(alpine_optimizer)

    # define the output variable
    new_icnn_output_var = @variable(new_icnn_lp, z, base_name="output_var")

    # define the input variable
    new_icnn_input_var = @variable(new_icnn_lp, x[1:n], base_name="input_var")

    # Define objective function
    if obj_expr !== nothing
        # Call the custom objective function with the model variables
        expr = obj_expr(new_icnn_input_var, new_icnn_output_var)
        set_objective(new_icnn_lp, obj_sense, expr)
    end

    # define the constraints on input variable
    for i in 1:n
        @constraint(new_icnn_lp, new_icnn_input_var[i] >= bounds_matrix[1, i])  # Lower bound constraint
        @constraint(new_icnn_lp, new_icnn_input_var[i] <= bounds_matrix[2, i])  # Upper bound constraint
    end

    # @constraint(new_icnn_lp, x <= extreme_points[2])
    # @constraint(new_icnn_lp, x >= extreme_points[1])

    # add variables, constraints, and an objective function to the icnn_lp model to account for the generated ICNN.
    ICNN_incorporate!(new_icnn_lp, filepath, new_icnn_output_var, new_icnn_input_var...)

    # Constraint 1: ai >= 0
    @variable(new_icnn_lp, a[1:num_extremes] >= 0)  

    # Constraint 2: x = sum(ai * extreme_points_i)
    # @constraint(new_icnn_lp, constraint_2, new_icnn_input_var == sum(a[i] * extreme_points[:,i] for i in 1:n))
    for j in 1:n
        @constraint(new_icnn_lp, new_icnn_input_var[j] == sum(a[i] * extreme_points[j, i] for i in 1:num_extremes))
    end

    # Constraint 3: z_k <= sum(ai * forwardpass(icnn_lp, extreme_points_i))
    extreme_point_values = [forwardpass(icnn_lp, extreme_points[:, i]) for i in 1:num_extremes]
    @constraint(new_icnn_lp, new_icnn_output_var <= sum(a[i] * extreme_point_values[i] for i in 1:num_extremes))

    # Consitrait 4: sum(ai) = 1
    @constraint(new_icnn_lp, constraint_4, sum(a) == 1)

    print(new_icnn_lp)

    return new_icnn_lp

end
