
function _add_envelope_constraints(icnn_lp, new_icnn_lp, box)

    bounds_matrix = generate_bounds_matrix(box)
    n = box.n  # Number of dimensions
    extreme_points = generate_extreme_combinations(bounds_matrix)
    num_extremes = size(extreme_points, 2)

    # define the icnn model
    new_icnn_lp = copy(new_icnn_lp)
    set_optimizer(new_icnn_lp, alpine_optimizer)

    # define the output variable
    new_icnn_output_var = new_icnn_lp[:z]

    # define the input variable
    new_icnn_input_var = new_icnn_lp[:x]

    # # Define objective function
    # if obj_expr !== nothing
    #     # Call the custom objective function with the model variables
    #     expr = obj_expr(new_icnn_input_var, new_icnn_output_var)
    #     set_objective(new_icnn_lp, obj_sense, expr)
    # end

    # define the constraints on input variable
    for i in 1:n
        @constraint(new_icnn_lp, new_icnn_input_var[i] >= bounds_matrix[1, i])  # Lower bound constraint
        @constraint(new_icnn_lp, new_icnn_input_var[i] <= bounds_matrix[2, i])  # Upper bound constraint
    end

    # Constraint 1: ai >= 0
    @variable(new_icnn_lp, a[1:num_extremes] >= 0)

    # Constraint 2: x = sum(ai * extreme_points_i)
    # @constraint(new_icnn_lp, constraint_2, new_icnn_input_var == sum(a[i] * extreme_points[:,i] for i in 1:n))
    for j in 1:n
        @constraint(new_icnn_lp, new_icnn_input_var[j] == sum(a[i] * extreme_points[j, i] for i in 1:num_extremes))
    end

    # Constraint 3: z_k <= sum(ai * forwardpass(icnn_lp, extreme_points_i))
    extreme_point_values = [forwardpass(icnn_lp, extreme_points[:, i]) for i in 1:num_extremes]
    @constraint(new_icnn_lp, constraint_3, new_icnn_output_var <= sum(a[i] * extreme_point_values[i] for i in 1:num_extremes))

    # Consitrait 4: sum(ai) = 1
    @constraint(new_icnn_lp, constraint_4, sum(a) == 1)

    print(new_icnn_lp)

    return new_icnn_lp

end

function generate_relaxation(icnn_lp, root_icnn_lp, boxes_list)
    new_icnn_lp_list = Array{JuMP.Model}(undef, length(boxes_list))

    for i in eachindex(boxes_list)
        current_box = boxes_list[i]
        new_icnn_lp = _add_envelope_constraints(icnn_lp, root_icnn_lp, current_box) 
        new_icnn_lp_list[i] = new_icnn_lp
        println("Model $i created successfully")
    end

    return new_icnn_lp_list
end