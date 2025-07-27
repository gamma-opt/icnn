using JuMP

mutable struct NodeOptResult
    model_index::Int
    model_status::MOI.TerminationStatusCode
    obj_optimal::Float64
    x_optimal::Union{Vector{Float64}, Vector{Vector{Float64}}, Nothing}
    z_optimal::Union{Float64, Vector{Float64}}
    a_values::Any  # Optional, can be removed if not needed
    icnn_z_optimal::Union{Float64, Vector{Float64}}
    gap::Float64
    is_feasible::Bool
    is_pruned::Bool
end

mutable struct Box
    n::Int
    lb::Vector
    ub::Vector
end

mutable struct TreeStatus
    obj_lb::Float64
    x_optimal::Union{Vector{Float64}, Nothing}
    bounds_to_branch::Vector{Tuple{Box, Int}}  # (box, dimension) pairs
    all_pruned::Bool
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
            if haskey(model, :a)
                # If 'a' is present, extract its values
                a_values = value.(model[:a])
            else
                # If 'a' is not present, set it to nothing or an empty vector
                a_values = nothing
            end
            icnn_z_optimal = forwardpass(icnn_lp, x_optimal)
            gap = abs(z_optimal - icnn_z_optimal)/abs(icnn_z_optimal) 
            is_feasible = true
            if gap > gap_tol
                is_feasible = false
                println("Warning: Gap between z_optimal and icnn_z_optimal is greater than $gap_tol, the envelope relaxation needs to be tightened")
            end
            is_pruned = false  # Default pruning status, can be updated later
                        
            # Store all data needed
            push!(results, NodeOptResult(
                i,
                termination_status(model),
                obj_optimal,
                x_optimal,
                z_optimal,
                a_values,
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
                NaN,
                nothing,
                NaN,
                nothing,
                NaN,
                NaN,
                false,
                true  # prune by infeasibility
            ))
            println("Optimisation did not reach optimality")
        end
    end

    return results
    
end

function solve_node_models_multi(model_list, icnn_lp_list; gap_tol = 0.01)

    results = []

    for i in eachindex(model_list)
        model = model_list[i]
        set_silent(model)
        println("\n----- Optimising Model $i -----")

        optimize!(model)
        
        println("Status: ", termination_status(model))
        
        if termination_status(model) == MOI.OPTIMAL
            obj_optimal = objective_value(model)
            x_optimal = [value.(x_group) for x_group in model[:x]]
            z_optimal = value.(model[:z])
            if haskey(model, :a)
                # If 'a' is present, extract its values
                a_values = value.(model[:a])
            else
                # If 'a' is not present, set it to nothing or an empty vector
                a_values = nothing
            end
            icnn_z_optimal = [forwardpass(icnn_lp_list[i], x) for (i, x) in enumerate(x_optimal)]
            # Compute the relative gap for each dimension and take the maximum as the overall gap
            gap = maximum(abs.(z_optimal .- icnn_z_optimal) ./ (abs.(icnn_z_optimal) .+ eps()))
            is_feasible = true
            if gap > gap_tol
                is_feasible = false
                println("Warning: Gap between z_optimal and icnn_z_optimal is greater than $gap_tol, the envelope relaxation needs to be tightened")
            end
            is_pruned = false  # Default pruning status, can be updated later
                        
            # Store all data needed
            push!(results, NodeOptResult(
                i,
                termination_status(model),
                obj_optimal,
                x_optimal,
                z_optimal,
                a_values,
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
                NaN,
                nothing,
                NaN,
                nothing,
                NaN,
                NaN,
                false,
                true  # prune by infeasibility
            ))
            println("Optimisation did not reach optimality")
        end
    end

    return results
    
end

function process_results(results::Vector{Any}, tree_status::TreeStatus)
     
    box_tuples = tree_status.bounds_to_branch
    obj_lb = tree_status.obj_lb
    x_optimal = tree_status.x_optimal

    bounds_to_branch = Tuple{Box, Int}[]
    
    # Process each result
    for i in eachindex(results)
        result = results[i]
        current_box, current_dimension = box_tuples[i] 
        println("\nProcessing result for model $(result.model_index) with box: $current_box, dimension: $current_dimension")
        
        # Only process if not already pruned
        if !result.is_pruned    
            if result.is_feasible
                result.is_pruned = true # prune by optimality
                if result.obj_optimal > obj_lb
                    obj_lb = max(result.obj_optimal, obj_lb)
                    println("...Updated lower bound: ", obj_lb)
                    x_optimal = result.x_optimal # store current best solution
                end
            else
                if result.obj_optimal < obj_lb
                    result.is_pruned = true  # prune by bound as ub < lb
                    println("...Current branch is pruned by bound")
                else
                    # TODO variable slection
                    #      - current_dimension could be changed based on some logic
                    #      - for nodes stemmed from the same box, they should be branched on the same dimension
                    #      - a trivial approach: cycles through dimensions
                    current_dimension = current_dimension < current_box.n ? current_dimension + 1 : 1
                    push!(bounds_to_branch, (current_box, current_dimension))
                    println("...Further branching on box $current_box with dimension $current_dimension is required")
                
                end
            end
        else
            println("...Current branch on box $current_box with dimension $current_dimension is pruned by infeasibility")
        end
    end
    
    # Check if all models are pruned
    all_pruned = all(r -> r.is_pruned, results)
    
    println("\nSummary:")
    if all_pruned
        println("\nAll nodes are pruned, no further branching needed")
        if !isnothing(x_optimal)
            println("\nOptimal solution found:")
            println("      x values: ", x_optimal)
            println("solution value: ", obj_lb)
        else
            println("\nNo feasible solution found")
            println("lower bound remains: ", obj_lb)
        end
    else
        println("\nFurther branching on $bounds_to_branch is required")
        println("\nCurrent lower bound: ", obj_lb)
    end

    tree_status.obj_lb = obj_lb
    tree_status.x_optimal = x_optimal
    tree_status.bounds_to_branch = bounds_to_branch
    tree_status.all_pruned = all_pruned
    
    return tree_status
end