function create_nn_data_3way(x, y; train_split=0.7, val_split=0.15, test_split=0.15)
    @assert train_split + val_split + test_split ≈ 1.0 "Splits must sum to 1.0"
    
    data = (x=x, y=y) 
    
    # First split: Separate test set
    train_val_data, test_data = Flux.splitobs(data, at = train_split + val_split) 
    
    # Second split: Separate train and validation sets from the remaining data
    adjusted_split = train_split / (train_split + val_split) 
    train_data, val_data = Flux.splitobs(train_val_data, at = adjusted_split)

    # Return named tuples for easy access, converting to Float32
    return (
        train = (x = Float32.(train_data.x), y = Float32.(train_data.y)),
        val = (x = Float32.(val_data.x), y = Float32.(val_data.y)),
        test = (x = Float32.(test_data.x), y = Float32.(test_data.y))
    )
end

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
<<<<<<< HEAD
        iteration += 1
=======
        # Increment and print iteration counter and current tree level
        iteration += 1

>>>>>>> multi_surrogate
        # Branch on the next box
        tree_status.bounds_to_branch = branch_box(tree_status.bounds_to_branch)
        
        # Generate new relaxation models for each branched box
        new_icnn_lp_list = generate_relaxation(icnn_lp, root_icnn_lp, tree_status)
        
        # Solve the new models
        # results = solve_node_models(new_icnn_lp_list, icnn_lp)
        results = icnn_lp isa Vector ?
            solve_node_models_multi(new_icnn_lp_list, icnn_lp) :
            solve_node_models(new_icnn_lp_list, icnn_lp)
        
        # Process the results and update the tree status
        tree_status = process_results(results, tree_status)        
    end
<<<<<<< HEAD
    
=======

>>>>>>> multi_surrogate
    println("\nTerminated at iteration #$iteration in $(round(time() - start_time, digits=4)) seconds")
end