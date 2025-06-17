mutable struct Box
    n::Int
    lb::Vector
    ub::Vector
end

function generate_bounds_matrix(box::Box; mid_dimension=nothing, bound_type=:lower)

    dimensions = box.n
    bounds_matrix = zeros(2, dimensions)
    
    # Fill the matrix with the bounds
    for i in 1:dimensions
        bounds_matrix[1, i] = box.lb[i]  # Lower bound for dimension i
        bounds_matrix[2, i] = box.ub[i]  # Upper bound for dimension i
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

function _bounds_matrix_to_box(bounds_matrix)
    dimensions = size(bounds_matrix, 2)
    lb = bounds_matrix[1, :]  # First row contains lower bounds
    ub = bounds_matrix[2, :]  # Second row contains upper bounds
    return Box(dimensions, lb, ub)
end

function branch_box(box; branch_dimension)
    # Generate bounds matrix with upper bound set to midpoint
    bounds_mid_upper = generate_bounds_matrix(box, mid_dimension=branch_dimension, bound_type=:upper)
    
    # Generate bounds matrix with lower bound set to midpoint  
    bounds_mid_lower = generate_bounds_matrix(box, mid_dimension=branch_dimension, bound_type=:lower)
    
    # Convert both bounds matrices to Box objects
    box_upper = _bounds_matrix_to_box(bounds_mid_upper)
    box_lower = _bounds_matrix_to_box(bounds_mid_lower)
    
    return [box_upper, box_lower]
end