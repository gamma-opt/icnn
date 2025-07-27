
function _add_envelope_constraints(icnn_lp, new_icnn_lp, box)

    bounds_matrix = generate_bounds_matrix(box)
    n = box.n  # Number of dimensions
    extreme_points = generate_extreme_combinations(bounds_matrix)
    num_extremes = size(extreme_points, 2)

    # define the icnn model
    new_icnn_lp = copy(new_icnn_lp)
    set_optimizer(new_icnn_lp, Gurobi.Optimizer)
    set_silent(new_icnn_lp)

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

    # print(new_icnn_lp)

    return new_icnn_lp
end

function _add_envelope_constraints_multi(icnn_lp_list, root_icnn_lp, box)

    # define the icnn model
    new_icnn_lp = copy(root_icnn_lp)
    set_optimizer(new_icnn_lp, Gurobi.Optimizer)

    # define the output/input variable
    new_icnn_output_var = new_icnn_lp[:z]
    new_icnn_input_var = new_icnn_lp[:x]

    dims = length(new_icnn_input_var)
    dims_eachx = map(length, new_icnn_input_var)
    
    
    bounds_matrix = generate_bounds_matrix(box)
    extreme_points_by_dim = Dict(
        dim => generate_extreme_combinations(bounds_matrix[:, sum(dims_eachx[1:dim-1]) + 1 : sum(dims_eachx[1:dim])]) 
        for dim in 1:dims
    )

    # define the constraints on input variable
    count = 0
    for dim in 1:dims, j in 1:dims_eachx[dim]
        count += 1
        @constraint(new_icnn_lp, bounds_matrix[1, count] <= new_icnn_input_var[dim][j] <= bounds_matrix[2, count])
    end

    # Constraint 1: ai >= 0, now 2D: a[extreme_point_index, dim_index]
    @variable(new_icnn_lp, a[dim=1:dims, ext=1:size(extreme_points_by_dim[dim], 2)] >= 0)

    # Constraint 2: x = sum(ai * extreme_points_i)
    for dim in 1:dims
        for j in 1:length(new_icnn_input_var[dim])
            @constraint(new_icnn_lp, new_icnn_input_var[dim][j] == sum(a[dim,i] * extreme_points_by_dim[dim][j, i] for i in 1:size(extreme_points_by_dim[dim], 2)))
        end
    end

    # Constraint 3: z_k <= sum(ai * forwardpass(icnn_lp, extreme_points_i))
    extreme_point_values = Dict{Int, Any}()
    for dim in 1:dims
        extreme_point_values[dim] = [forwardpass(icnn_lp_list[dim], extreme_points_by_dim[dim][:, i]) for i in 1:size(extreme_points_by_dim[dim], 2)]
    end

    for dim in 1:dims
        @constraint(new_icnn_lp, new_icnn_output_var[dim] <= sum(a[dim,i] * extreme_point_values[dim][i] for i in 1:size(extreme_points_by_dim[dim], 2)))
    end

    # Constraint 4: sum(ai) = 1
    for dim in 1:dims
        @constraint(new_icnn_lp, sum(a[dim, i] for i in 1:size(extreme_points_by_dim[dim], 2)) == 1)
    end

    return new_icnn_lp
end

function generate_relaxation(icnn_lp, root_icnn_lp, tree_status::TreeStatus)
    boxes_list = [box for (box, _) in tree_status.bounds_to_branch]
    new_icnn_lp_list = Array{JuMP.Model}(undef, length(boxes_list))

    for i in eachindex(boxes_list)
        current_box = boxes_list[i]
        new_icnn_lp = _add_envelope_constraints(icnn_lp, root_icnn_lp, current_box) 
        new_icnn_lp_list[i] = new_icnn_lp
        println("Model $i created successfully")
    end

    return new_icnn_lp_list
end