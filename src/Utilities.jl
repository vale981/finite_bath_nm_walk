includet("./WalkModel.jl")

module Utilities
export plot_analytic_phase_diagram

using ..WalkModel
import Plots: heatmap, hline!, vline!

function plot_analytic_phase_diagram(grid_N::Integer=50, α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2);
    num_bath_modes::Integer=200, bath_discretization::Function=exponential_energy_distribution, coupling_strength::Real=0.01, Δ::Real=5, ε_min::Real = .01)
    displacement = Array{Float64}(undef, grid_N, grid_N)

    vv = 1
    αs = collect(LinRange(α_limits..., grid_N))
    us = collect(LinRange(u_limits..., grid_N))

    Threads.@threads for i in 1:grid_N
        for j in 1:grid_N
            α = αs[j]
            u = us[i]
            sd = OhmicSpectralDensity((Δ * vv), coupling_strength, α)
            ε, g = bath_discretization(sd, num_bath_modes, ε_min * vv)
            params = ModelParameters(v=vv, u=u, ε=ε, g=g, sw_approximation=true)

            displacement[i, j] = analytic_time_averaged_displacement(params)
        end
    end

    p = heatmap(αs, us, displacement, xlabel=raw"$α$", ylabel=raw"$u$")
    vline!([1], label=false, color=:white)
    hline!([1], label=false, color=:white)

    p
end
end
