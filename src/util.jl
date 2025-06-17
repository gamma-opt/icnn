
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
