
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
    for i in eachindex(input_values)
        fix(jump_model[:x][i], input_values[i]; force = true)
    end
    set_silent(jump_model)
    optimize!(jump_model)

    result = value.(jump_model[:z])
    
    # Unfix each element individually
    for i in eachindex(input_values)
        unfix(jump_model[:x][i])
    end

    return result
end

function branch_and_bound(icnn_lp, root_icnn_lp, tree_status::TreeStatus)
    # Start with the root node
    new_icnn_lp_list = [root_icnn_lp]
    iteration = 0
    start_time = time()
    # Process the tree status to branch on boxes
    while !tree_status.all_pruned
        iteration += 1
        # Branch on the next box
        tree_status.bounds_to_branch = branch_box(tree_status.bounds_to_branch)
        
        # Generate new relaxation models for each branched box
        new_icnn_lp_list = generate_relaxation(icnn_lp, root_icnn_lp, tree_status)
        
        # Solve the new models
        results = solve_node_models(new_icnn_lp_list, icnn_lp)
        
        # Process the results and update the tree status
        tree_status = process_results(results, tree_status)
    end
    
    println("\nTerminated at iteration #$iteration in $(round(time() - start_time, digits=4)) seconds")
end