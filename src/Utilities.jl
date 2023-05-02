includet("./WalkModel.jl")

module Utilities
export plot_analytic_phase_diagram
export plot_phase_diagram
export plot_analytic_phase_diagram_born_v_exact
export plot_overview
export scan_setproperties

using ..WalkModel
import Plots: heatmap, hline!, vline!
import Statistics: mean
using Plots
import ConstructionBase: setproperties
using Accessors: @set, @reset

function plot_analytic_phase_diagram(grid_N::Integer=50; α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2),
    num_bath_modes::Integer=200, bath_discretization::Function=exponential_energy_distribution, coupling_strength::Real=0.01, ω_c::Real=1, ε_min::Real=0, integrand=integrand_diagonalization, T::Real=0, normalize::Bool = true)
    displacement = Array{Float64}(undef, grid_N, grid_N)

    vv = 1
    αs = collect(LinRange(α_limits..., grid_N))
    us = collect(LinRange(u_limits..., grid_N))

    min_e = Inf
    max_g = 0
    Threads.@threads for i in 1:grid_N
        for j in 1:grid_N
            α = αs[j]
            u = us[i]
            sd = OhmicSpectralDensity((ω_c), coupling_strength, α)
            ε, g = bath_discretization(sd, num_bath_modes, normalize)

            if ε[begin] < min_e
                min_e = ε[begin]
            end

            if g[begin] > max_g
                max_g = g[begin]
            end

            ε .-= ε_min
            params = ModelParameters(v=vv, u=u, ε=ε, g=g, sw_approximation=true)
            displacement[i, j] = if T > 0
                analytic_time_averaged_displacement(T, params, integrand)
            else
                analytic_time_averaged_displacement(params, integrand)
            end
        end
    end

    @show maximum(displacement)

    p = heatmap(αs, us, displacement, xlabel=raw"$α$", ylabel=raw"$u$", title=raw"$\langle m\rangle$")
    vline!([1], label=false, color=:white)
    hline!([1], label=false, color=:white)

    p
end

function plot_analytic_phase_diagram_born_v_exact(grid_N::Integer=50, α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2);
    num_bath_modes::Integer=200, bath_discretization::Function=exponential_energy_distribution, coupling_strength::Real=0.01, ω_c::Real=1, ε_min::Real=0, integrand=integrand_diagonalization)
    displacement = Array{Float64}(undef, grid_N, grid_N)

    vv = 1
    αs = collect(LinRange(α_limits..., grid_N))
    us = collect(LinRange(u_limits..., grid_N))

    Threads.@threads for i in 1:grid_N
        for j in 1:grid_N
            α = αs[j]
            u = us[i]
            sd = OhmicSpectralDensity((ω_c * vv), coupling_strength, α)
            ε, g = bath_discretization(sd, num_bath_modes)
            ε .-= ε_min
            params = ModelParameters(v=vv, u=u, ε=ε, g=g, sw_approximation=true)

            ex_disp =  abs(analytic_time_averaged_displacement(params, integrand_diagonalization))
            ex_disp_norm = if ex_disp == 0
                1
            else
                ex_disp
            end

            displacement[i, j] = abs(ex_disp - analytic_time_averaged_displacement(params, integrand_born)) / ex_disp_norm
        end
    end

    @show mean(displacement)
    p = heatmap(αs, us, displacement, xlabel=raw"$α$", ylabel=raw"$u$", title=raw"$(\langle m\rangle - \langle m_\mathrm{Born}\rangle)/\langle m\rangle$")
    vline!([1], label=false, color=:white)
    hline!([1], label=false, color=:white)

    p
end

function plot_overview(p::ExtendedModelParameters, T::Real, k::Real = 0)
    params = ModelParameters(p)
    sol = WalkSolution(k, params)

    plot(t->mean_displacement(t, params), 0.1, T, label=raw"$\langle m\rangle$", xlabel="t", title="u=$(p.u), α=$(p.spectral_density.α), N=$(p.N)")
    plot!(t->analytic_time_averaged_displacement(t, params), label=raw"$\overline{\langle m\rangle}$")
    hline!(t->analytic_time_averaged_displacement(params), label=raw"$\overline{\langle m\rangle}(T=\infty)$")

    plot!(t->a_weight(t, sol) * 2π, label="\$\\rho_A(k=$(k))\$")
end

plot_overview(params::ExtendedModelParameters, rest...) = plot_overview(ModelParameters(params), rest...)

function scan_setproperties(strct::Any; kwargs...)
      fields = collect(keys(kwargs))
      vals = collect(values(kwargs))

      [setproperties(strct; Dict(zip(fields, current_values))...) for current_values in (Iterators.product(vals...))]
end


function plot_phase_diagram(params::ExtendedModelParameters, grid_N::Integer=50; α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2), shift_k::Real=0)
    displacement = Array{Float64}(undef, grid_N, grid_N)

    αs = collect(LinRange(α_limits..., grid_N))
    us = collect(LinRange(u_limits..., grid_N))

    Threads.@threads for i in 1:grid_N
        for j in 1:grid_N
            α = αs[j]
            u = us[i]



            current_params = @set params.spectral_density.α = α
            @reset current_params.u = u

            current_params = auto_shift_bath(current_params, shift_k)
            displacement[i, j] = mean_displacement(recurrence_time(current_params) * .95, current_params |> ModelParameters)
        end
    end

    @show maximum(displacement)

    p = heatmap(αs, us, displacement, xlabel=raw"$α$", ylabel=raw"$u$", title=raw"$\langle m\rangle$")
    vline!([1], label=false, color=:white)
    hline!([1], label=false, color=:white)

    p
end

end
