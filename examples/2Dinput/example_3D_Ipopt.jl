using JuMP
using Ipopt

lb = [-1.0, -1.0]
ub = [0.0, 0.0]

full_model = Model(Ipopt.Optimizer)
@variable(full_model, x[1:2])
@variable(full_model, z)
@constraints(full_model, begin
    lb[1] <= x[1] <= ub[1]
    lb[2] <= x[2] <= ub[2]
end)
@constraint(full_model, z >= 1.0)
@NLconstraint(full_model, z == x[1]^2 + x[2]^2)
@objective(full_model, Max, x[1])
optimize!(full_model)

println("Optimal x values: ", value.(full_model[:x]))
println("Optimal objective value: ", objective_value(full_model))