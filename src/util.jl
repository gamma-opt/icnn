"""
    generate_bounds_matrix(dimensions, bounds_vectors...; mid_dimension=nothing, bound_type=:lower)

Generate a bounds matrix for a given number of dimensions, where each column represents the lower and upper bounds
for a specific dimension. Optionally, modify the bound of a specific dimension to its midpoint.

# Arguments
- `dimensions::Int`: The number of dimensions.
- `bounds_vectors::Vararg{Tuple{Float64, Float64}}`: A variable number of 2-element tuples, each representing the lower and upper bounds for a dimension.
- `mid_dimension::Union{Int, Nothing}`: (Optional) The dimension whose bound should be modified to its midpoint. Defaults to `nothing`.
- `bound_type::Symbol`: (Optional) Specifies whether to modify the lower (`:lower`) or upper (`:upper`) bound of the specified dimension. Defaults to `:lower`.

# Returns
- `bounds_matrix::Matrix{Float64}`: A 2x`dimensions` matrix where the first row contains the lower bounds and the second row contains the upper bounds.

"""

function generate_bounds_matrix(dimensions, bounds_vectors...; mid_dimension=nothing, bound_type=:lower)
    # Check that we have the correct number of bounds vectors
    if length(bounds_vectors) != dimensions
        error("Must provide bounds vectors for each of the $dimensions dimensions")
    end
    
    # Check that each bounds vector has exactly 2 elements (lb and ub)
    for (i, bounds) in enumerate(bounds_vectors)
        if length(bounds) != 2
            error("Bounds vector for dimension $i must have exactly 2 elements (lb and ub)")
        end
    end
    
    # Initialise the bounds matrix (2 rows, n columns)
    bounds_matrix = zeros(2, dimensions)
    
    # Fill the matrix with the bounds
    for i in 1:dimensions
        bounds_matrix[1, i] = bounds_vectors[i][1]  # Lower bound for dimension i
        bounds_matrix[2, i] = bounds_vectors[i][2]  # Upper bound for dimension i
    end
    
    # Modify a specific dimension's bound to its halfway point if requested
    if mid_dimension !== nothing
        if mid_dimension < 1 || mid_dimension > dimensions
            error("mid_dimension must be between 1 and $dimensions")
        end
        
        # Calculate the midpoint
        lb = bounds_matrix[1, mid_dimension]
        ub = bounds_matrix[2, mid_dimension]
        mid_value = lb + (ub - lb) / 2
        
        # Update the specified bound with the midpoint
        if bound_type == :lower
            bounds_matrix[1, mid_dimension] = mid_value
        elseif bound_type == :upper
            bounds_matrix[2, mid_dimension] = mid_value
        else
            error("bound_type must be either :lower or :upper")
        end
    end
    
    return bounds_matrix
end

"""
    generate_extreme_combinations(bounds_matrix)

Generate all extreme combinations of points based on the given bounds matrix. Each combination corresponds to a corner
of the hyper-rectangle defined by the bounds.

# Arguments
- `bounds_matrix::Matrix{Float64}`: A 2xN matrix where the first row contains the lower bounds and the second row contains the upper bounds for each dimension.

# Returns
- `combinations::Matrix{Float64}`: An N x 2^N matrix where each column represents an extreme point (corner) of the hyper-rectangle.

"""

function generate_extreme_combinations(bounds_matrix)
    dimensions = size(bounds_matrix, 2)
    num_combinations = 2^dimensions
    
    # Create a matrix to store all combinations
    # Each column will be one extreme point
    combinations = zeros(dimensions, num_combinations)
    
    # Generate all combinations of extreme points
    for i in 0:(num_combinations-1)
        binary = digits(i, base=2, pad=dimensions) # Convert to binary representation
        
        for dim in 1:dimensions
            # If the bit is 0, use lower bound; if 1, use upper bound
            idx = binary[dim] + 1 # 1-based indexing (1 for lb, 2 for ub)
            combinations[dim, i+1] = bounds_matrix[idx, dim]
        end
    end
    
    return combinations
end


"""
    forwardpass(jump_model, input_values)

Performs a forward pass through the given jump model with the specified input values.

# Arguments
- `jump_model::JuMP.Model`: The JuMP model to be used for the forward pass.
- `input_values::Vector`: A vector of input values to be used in the forward pass.

# Returns
- `result::Vector`: The result of the forward pass, which is a vector of output values.

"""

function forwardpass(jump_model, input_values)
    # Fix each element individually
    for i in 1:2
        fix(jump_model[:x][i], input_values[i]; force = true)
    end
    optimize!(jump_model)

    result = value.(jump_model[:z])
    
    # Unfix each element individually
    for i in 1:2
        unfix(jump_model[:x][i])
    end

    return result
end

"""
    optimise_and_print_results(model_list, icnn_lp)

Optimises a list of models and prints the results.

# Arguments
- `model_list::Vector`: A list of models to be optimised.
- `icnn_lp`: A specific parameter or object used during the optimisation process.

"""

function optimise_and_print_results(model_list, icnn_lp)

    results = []

    for i in eachindex(model_list)
        model = model_list[i]
        println("\n----- Optimising Model $i -----")

        optimize!(model)
        
        println("Status: ", termination_status(model))
        
        if termination_status(model) == MOI.OPTIMAL
            obj_optimal = objective_value(model)
            x_optimal = value.(model[:x])
            z_optimal = value(model[:z])
            a_values = value.(model[:a])
            icnn_z_optimal = forwardpass(icnn_lp, x_optimal)
            
            # Store all data needed for plotting
            push!(results, (
                model_index = i,
                obj_optimal = obj_optimal,
                x_optimal = x_optimal,
                z_optimal = z_optimal,
                a_values = a_values,
                icnn_z_optimal = icnn_z_optimal
            ))
            
            println("Objective value: ", obj_optimal)
            println("x value: ", x_optimal)
            println("z value: ", z_optimal)
            println("icnn(x) value: ", icnn_z_optimal)
            println("a values:")
            for j in eachindex(a_values)
                println("  a[$j] = ", a_values[j])
            end
        else
            println("Optimisation did not reach optimality.")
        end
    end

    return results
    
end