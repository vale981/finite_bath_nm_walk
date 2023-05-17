includet("./WalkModel.jl")

module Utilities
export plot_analytic_phase_diagram
export plot_phase_diagram
export plot_analytic_phase_diagram_born_v_exact
export plot_overview
export plot_overview_windowed
export plot_ρ_A
export scan_setproperties
export plot_A_overlap
export ρ_A_k_overview
export @parametrize_properties

using ..WalkModel
using LaTeXStrings
import Plots: heatmap, hline!, vline!
import Statistics: mean
using Plots
import ConstructionBase: setproperties
using Accessors: @set, @reset
using LinearAlgebra

function plot_analytic_phase_diagram(grid_N::Integer=50; α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2),
    num_bath_modes::Integer=200, bath_discretization::Function=exponential_energy_distribution, coupling_strength::Real=0.01, ω_c::Real=1, ε_min::Real=0, integrand=integrand_diagonalization, T::Real=0, normalize::Bool=true)
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

            ex_disp = abs(analytic_time_averaged_displacement(params, integrand_diagonalization))
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

function plot_overview(p::ExtendedModelParameters, T::Real, k::Real=0)
    params = ModelParameters(p)
    sol = WalkSolution(k, params)

    plot(t -> mean_displacement(t, params), 0.1, T, label=L"$\langle m(t)\rangle$", xlabel=L"$t$", title=L"$u=%$(p.u)$, $\alpha=%$(p.spectral_density.α)$, $N=%$(p.N)$")
    plot!(t -> analytic_time_averaged_displacement(t, params), label=L"$\langle m\rangle$ running")
    hline!([analytic_time_averaged_displacement(params)], label=L"$\langle m\rangle$")

    plot!(t -> a_weight(t, sol) * 2π, label=L"$\rho_A(k=%$(k))$")
end

function plot_overview_windowed(p::ExtendedModelParameters, T::Real, k::Real=0)
    params = ModelParameters(p)
    sol = WalkSolution(k, params)

    τ = 2.5 / (decay_rate(p, π))
    plot(t -> mean_displacement(t, params), 0.1, T, label=L"$\langle m(t)\rangle$", xlabel=L"$t$", title=L"$u=%$(p.u)$, $\alpha=%$(p.spectral_density.α)$, $N=%$(p.N)$")
    hline!([analytic_time_averaged_displacement(params)], label=L"$\langle m\rangle$", linestyle=:dash)
    hline!([analytic_time_averaged_displacement(τ, 0.9 * recurrence_time(p), params)], label=L"$\langle m\rangle$ (windowed)")

    plot!(t -> a_weight(t, sol) * 2π, label=L"$\rho_A(k=%$(k))$")
    plot!(t -> ρ_A_mean(τ, t, sol) * 2π, label=L"$\rho_A(k=%$(k))$ windowed average")
end


function plot_ρ_A(p::ExtendedModelParameters, T::Real, k::Real=0)
    params = ModelParameters(p)
    sol = WalkSolution(k, params)

    plot(t -> a_weight(t, sol) * 2π, 0, T, xlabel="t", label="\$\\rho_A(k=$(k))\$")
end

plot_overview(params::ExtendedModelParameters, rest...) = plot_overview(ModelParameters(params), rest...)

function scan_setproperties(strct::Any; kwargs...)
    fields = collect(keys(kwargs))
    vals = collect(values(kwargs))

    [setproperties(strct; Dict(zip(fields, current_values))...) for current_values in (Iterators.product(vals...))]
end


function splice_accessor(symbol, arg1)
    Expr(:., symbol, QuoteNode(arg1))
end

function splice_accessor(symbol, arg1, arg2)
    if isa(arg1, Expr)
        Expr(:., splice_accessor(symbol, arg1.args...), arg2)
    else
        Expr(:., Expr(:., symbol, QuoteNode(arg1)), arg2)
    end
end

macro parametrize_properties(strct::Any, args...)
    names = map(_ -> gensym("arg"), args)
    args = map(args) do spec
        acc_args = if isa(spec, Expr)
            spec.args
        else
            [spec]
        end
    end
    arg_tuples = map(zip(names, args)) do (name, spec)
        :($(name) = $(splice_accessor(esc(strct), spec...)))
    end
    function_args = Expr(:tuple, arg_tuples...)

    tmp = gensym("strct")
    assignments = map(zip(names, args)) do (name, arg)
        acc = splice_accessor(tmp, arg...)
        :($(tmp) = @set $(acc) = $(name))
    end

    Expr(:function, function_args, quote
        local $(tmp) = $(esc(strct))
        $(assignments...)

        $(tmp)
    end)
end


function plot_phase_diagram(params::ExtendedModelParameters, grid_N::Integer=50;
    α_limits::Tuple{Real,Real}=(0, 2), u_limits::Tuple{Real,Real}=(0, 2), shift_A::Bool=true, shift_k::Real=0,
    window::Bool=true, window_k::Real=π)
    displacement = Array{Float64}(undef, grid_N, grid_N)

    αs = collect(LinRange(α_limits..., grid_N))
    us = collect(LinRange(u_limits..., grid_N))

    dα = αs[2] - αs[1]
    du = us[2] - us[1]

    Threads.@threads for i in 1:grid_N
        for j in 1:grid_N
            α = αs[j] + dα/2
            u = us[i] + du/2


            current_params = @set params.spectral_density.α = α
            @reset current_params.u = u

            if shift_A
                current_params = auto_shift_bath(current_params, shift_k)
            end

            τ_end = recurrence_time(current_params) * 0.95
            if window
                # τ = 1 / (decay_rate(current_params, window_k))
                # if τ > τ_end
                #     @show τ_end, α, u, decay_rate(current_params, window_k),  current_params.ω_A
                #     error("Decay doesn't take place before recurrence.")
                # end
                displacement[i, j] = analytic_time_averaged_displacement(.5 * τ_end, τ_end, current_params |> ModelParameters)
            else
                #displacement[i, j] = mean_displacement(recurrence_time(current_params) * 0.95, current_params |> ModelParameters)
                displacement[i, j] = analytic_time_averaged_displacement(current_params |> ModelParameters)
            end
        end
    end

    @show maximum(displacement)

    p = heatmap(αs, us, displacement, xlabel=L"$\alpha$", ylabel=L"$u$", title=L"$\langle m\rangle$")
    vline!([1], label=false, color=:white)
    hline!([1], label=false, color=:white)

    p
end

function plot_A_overlap(params::ModelParameters, k::Real=0)
    H = hamiltonian(k, params)
    ψ_A = [1; zeros(num_bath_modes(params))]
    energies = eigvals(H)
    overlaps = (ψ_A' * eigvecs(H) .|> abs2)'
    bar(energies, overlaps, ylabel="Overlap with A", xlabel=L"$E$", label=false)
end

plot_A_overlap(params::ExtendedModelParameters, rest...) = plot_A_overlap(params |> ModelParameters, rest...)

function ρ_A_k_overview(α::Real, k_shift::Real, params::ExtendedModelParameters)
    params = auto_shift_bath(params, k_shift)
    @reset params.spectral_density.α = α
    plot(xlabel="t", title="\$α=$(params.spectral_density.α)\$, \$k_s=$(k_shift)\$, \$N=$(params.N)\$")
    for k in LinRange(0, π, 4)
        plot!(a_weight(params, k), 0, 1.2 * recurrence_time(params), label="\$ρ_A(k=$(round(k / π, sigdigits=2)) π)\$")
    end
    vline!([0.95 * recurrence_time(params)], label=false)
end

end
