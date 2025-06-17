mutable struct NodeOptResult
    model_index::Int
    model_status
    obj_optimal
    x_optimal
    z_optimal
    icnn_z_optimal
    gap
    is_feasible::Bool
    is_pruned::Bool
end

function solve_node_models(model_list, icnn_lp; gap_tol = 0.01)

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
            # a_values = value.(model[:a])
            icnn_z_optimal = forwardpass(icnn_lp, x_optimal)
            gap = abs(z_optimal - icnn_z_optimal)/abs(icnn_z_optimal) 
            is_feasible = true
            if gap > gap_tol
                is_feasible = false
                println("Warning: Gap between z_optimal and icnn_z_optimal is greater than $gap_tol, the envelope relaxation needs to be tightened")
            end
            is_pruned = false  # Default pruning status, can be updated later
                        
            # Store all data needed for plotting
            push!(results, NodeOptResult(
                i,
                termination_status(model),
                obj_optimal,
                x_optimal,
                z_optimal,
                # a_values,
                icnn_z_optimal,
                gap,
                is_feasible,
                is_pruned
            ))
            
            println("Objective value: ", obj_optimal)
            println("        x value: ", x_optimal)
            println("        z value: ", z_optimal)
            println("  icnn(x) value: ", icnn_z_optimal)
            # println("a values:")
            # for j in eachindex(a_values)
                # println("  a[$j] = ", a_values[j])
            # end
            println("           Gap: ", gap)
        else
            push!(results, NodeOptResult(
                i,
                termination_status(model),
                nothing,
                nothing,
                nothing,
                # nothing,
                nothing,
                nothing,
                nothing,
                true  # prune by infeasibility
            ))
            println("Optimisation did not reach optimality.")
        end
    end

    return results
    
end


function process_results(results, box_list, obj_lb, x_values)
    bounds_to_branch = Box[]
    
    # Process each result
    for i in eachindex(results)
        result = results[i]
        current_box = box_list[i] 
        println("\nProcessing result for model $(result.model_index) with box: ", current_box)
        
        # Only process if not already pruned
        if !result.is_pruned    
            if result.is_feasible
                result.is_pruned = true # prune by optimality
                if result.obj_optimal > obj_lb
                    obj_lb = max(result.obj_optimal, obj_lb)
                    println("Updated lower bound: ", obj_lb)
                    x_values = result.x_optimal # store current best solution
                end
            else
                if result.obj_optimal < obj_lb
                    result.is_pruned = true  # prune by bound as ub < lb
                    println("Current branch is pruned by bound")
                else
                    push!(bounds_to_branch, current_box)
                    println("Further branching on $current_box is required")
                end
            end
        end
    end
    
    # Check if all models are pruned
    all_pruned = all(r -> r.is_pruned, results)
    
    if all_pruned
        println("\nAll nodes are pruned, no further branching needed")
        if !isnothing(x_values)
            println("      x values: ", x_values)
            println("solution value: ", obj_lb)
        end
    else
        println("\nFurther branching on $bounds_to_branch is required")
        println("Current lower bound: ", obj_lb)
    end
    
    return obj_lb, x_values, bounds_to_branch, all_pruned
end