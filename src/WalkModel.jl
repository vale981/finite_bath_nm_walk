module WalkModel
"""An implementation of the non-markovian quantum walk in k-space with a discrete bath"""

export ModelParameters
export ExtendedModelParameters
export v
export hamiltonian
export solution
export WalkSolution
export Σ_A
export ρ_A
export ρ_A_mean
export ρ_A_mean_born
export dϕ
export analytic_time_averaged_displacement
export construct_residue_poly
export residues_poly
export OhmicSpectralDensity
export OhmicSpectralDensityIntegral
export linear_energy_distribution
export exponential_energy_distribution
export a_weight
export non_a_weight
export mean_displacement
export integrand_born
export integrand_diagonalization
export integrand_residue
export time_scale
export num_bath_modes
export Discretization
export LinearBathDiscretization
export ExponentialBathDiscretization
export discretization_name
export discretize_bath
export lamb_shift
export optimal_bath_shift
export auto_shift_bath
export recurrence_time
export minimal_N
export decay_rate

using Parameters
import LinearAlgebra: diagm, eigen, ⋅
import Cubature: hquadrature, hquadrature_v
import Statistics: mean
import SpecialFunctions: gamma, zeta
import Polynomials: Polynomial, fromroots, coeffs
import PolynomialRoots: roots
import Accessors: @set

abstract type BathDiscretization end
abstract type LinearBathDiscretization <: BathDiscretization end
abstract type ExponentialBathDiscretization <: BathDiscretization end
discretization_name(::Type{LinearBathDiscretization}) = "linear"
discretization_name(::Type{ExponentialBathDiscretization}) = "exponential"

"""The parameters required to simulate the model, agnostic of how they
are arrived at."""
@with_kw struct ModelParameters
    """Intracell hopping."""
    v::Real = 1

    """Ratio of inter to intracell hopping."""
    u::Real = 0.5

    """The site energy detuning."""
    ω::Real = 0.1

    """Bath energy levels."""
    ε::Vector{<:Real} = [1.0, 2.0]

    """Bath coupling strengths."""
    g::Vector{<:Real} = [1.0, 2.0]
    @assert length(g) == length(ε)

    """Whether the system should simulated in the Schrieffer-Wolff
       approximation."""
    sw_approximation::Bool = false

    """Energy of the A site"""
    ω_A::Real = 0
end



struct OhmicSpectralDensity
    ω_c::Real
    J::Real
    α::Real
end


(J::OhmicSpectralDensity)(ε::Real) =
    if ε < J.ω_c
        J.J * ε^J.α * ((J.α + 1) / J.ω_c^(J.α + 1))
    else
        0
    end

struct OhmicSpectralDensityIntegral
    ω_c::Real
    J::Real
    α::Real
end

OhmicSpectralDensityIntegral(J::OhmicSpectralDensity) = OhmicSpectralDensityIntegral(J.ω_c, J.J, J.α)

(J::OhmicSpectralDensityIntegral)(ε::Real) =
    if ε < J.ω_c
        J.J * ε^(J.α + 1) / J.ω_c^(J.α + 1)
    else
        J.J
    end

"""The parameters required to simulate the model, agnostic of how they
are arrived at.
Instead of specifying the bath spectrum and coupling directly, these values are computed later on."""
@with_kw struct ExtendedModelParameters
    """Intracell hopping."""
    v::Real = 1

    """Ratio of inter to intracell hopping."""
    u::Real = 0.5

    """The site energy detuning."""
    ω::Real = 0.1

    spectral_density::OhmicSpectralDensity = OhmicSpectralDensity(1, 0.01, 1)

    N::Integer = 10
    discretization::Type{<:BathDiscretization} = LinearBathDiscretization

    normalize::Bool = true

    """Whether the system should simulated in the Schrieffer-Wolff
       approximation."""
    sw_approximation::Bool = false

    ω_A::Real = 0
end

function ModelParameters(p::ExtendedModelParameters)
    ε, g = discretize_bath(p)

    ModelParameters(p.v, p.u, p.ω, ε, g, p.sw_approximation, p.ω_A)
end

function convert(::Type{ModelParameters}, p::ExtendedModelParameters)
    ModelParameters(p)
end

OhmicSpectralDensity(params::ExtendedModelParameters) = params.spectral_density
convert(::Type{OhmicSpectralDensity}, params::ExtendedModelParameters) = OhmicSpectralDensity(params)


"""Returns the number of bath states for the model."""
num_bath_modes(p::ModelParameters) = length(p.g)

"""
    v(k, v, u)

The complex ``k`` rependent hopping amplitude, where `v` is the
intracell hopping and `u` is the ratio of inter to intracell
hopping."""
v(k::Real, v::Real, u::Real)::Complex = v + u * v * exp(complex(0, k))
v(k::Real, params::ModelParameters)::Complex = v(k, params.v, params.u)
v(k::Real, params::ExtendedModelParameters)::Complex = v(k, params.v, params.u)

"""The winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
ϕ(args...)::Real = angle(v(args...))

"""The derivative of winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
dϕ(k::Real, u::Real)::Real = u * (u + cos(k)) / (u^2 + 1 + 2 * u * cos(k))
dϕ(k::Real, params::ModelParameters)::Real = dϕ(k, params.u)

"""
  hamiltonian(k, params)

Returns the model Hamiltonian at momentum ``k`` for the `params`.  The
basis is ``A, B, bath levels``.
"""
function hamiltonian(k::Real, params::ModelParameters)::Matrix{<:Number}
    v_complex = v(k, params)
    H_bath = diagm(0 => params.ε)

    if params.sw_approximation
        g = v_complex .* params.g / abs(v(0, params))

        [params.ω_A g'
            g H_bath]
    else
        V = [0 conj(v_complex)
            v_complex 0]

        H_AB = [params.ω_A 0; 0 params.ω]

        H_system_bath = [zeros(num_bath_modes(params))'
            (params.g)']

        [(V+H_AB) H_system_bath
            H_system_bath' H_bath]
    end
end


"""A structure holding the information for the dynamic solution of the
   quantum walk for a specific ``k``. Callable with a time parameter."""
struct WalkSolution
    vectors::Matrix{<:Complex}
    energies::Vector{<:Real}
    params::ModelParameters
end

solution(args...) = WalkSolution(args...)
function WalkSolution(k::Real, params::ModelParameters, m_0::Integer=0)
    H = hamiltonian(k, params)
    energies, vectors = eigen(H)
    energies = real.(energies)

    ψ_0 = if params.sw_approximation
        [exp(-1im * k * m_0); zeros(num_bath_modes(params))]
    else
        [exp(-1im * k * m_0); 0; zeros(num_bath_modes(params))]
    end

    coefficients = vectors' * ψ_0 / sqrt(2π)

    WalkSolution((coefficients' .* vectors), energies, params)
end

WalkSolution(k::Real, params::ExtendedModelParameters, m_0::Integer=0) = WalkSolution(k, params |> ModelParameters, m_0)

"""
   (WalkSolution)(t)

The solution at time ``t``."""
function (sol::WalkSolution)(t::Real)
    @inbounds sol.vectors * (exp.(complex.(0, -sol.energies * t)))
end

"""The ``\rho_{\bar{A}}(k, t)`` at time ``t`` for the specific solution `sol`."""
# non_a_weight(t::Real, sol::WalkSolution)::Real = (if sol.params.sw_approximation
#                                                       sol(t)[1:end]
#                                                   else
#                                                       sol(t)[2:end]
#                                                   end).|> abs2 |> sum
a_weight(t::Real, sol::WalkSolution)::Real = sol.vectors[1, :] ⋅ (exp.(complex.(0, -sol.energies * t))) |> abs2
function a_weight(p::ExtendedModelParameters, k::Real=0)
    sol = WalkSolution(k, p |> ModelParameters)
    return t -> a_weight(t, sol)
end

non_a_weight(t::Real, sol::WalkSolution)::Real = (1 / (2π) - abs2(sol(t)[1]))


function discretize_bath(scheme::Type{<:BathDiscretization}, J::OhmicSpectralDensity, N::Integer, normalize::Bool=true)
    xk, ε = find_nodes_and_energies(scheme, J, N)

    J_int = OhmicSpectralDensityIntegral(J)
    g = ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))

    if normalize
        g /= sum(abs.(g))
        g *= J.J
    end

    ε, sqrt.(g)
end
discretize_bath(p::ExtendedModelParameters) = discretize_bath(p.discretization, OhmicSpectralDensity(p), p.N, p.normalize)

function find_nodes_and_energies(::Type{LinearBathDiscretization}, J::OhmicSpectralDensity, N::Integer)
    Δ = J.ω_c
    xk = collect(LinRange(0, Δ, N + 1))
    ε = Δ * ((2 * collect(1:N) .- 1) / (2 * N))

    xk, ε
end

function find_nodes_and_energies(::Type{ExponentialBathDiscretization}, J::OhmicSpectralDensity, N::Integer)
    ω_c = -J.ω_c / log(1 / (2N))
    xk = -ω_c * log.(1 .- collect(0:N) / (N))
    ε = -ω_c * log.(1 .- (2 * collect(1:N) .- 1) / (2 * N))

    xk, ε
end


function mean_displacement(t::Real, params::ModelParameters, m_0::Integer=0)
    function integrand(ks, v)
        Threads.@threads for i = 1:length(ks)
            k = ks[i]
            sol = solution(k, params, m_0)
            @inbounds v[i] = dϕ(k, params) * non_a_weight(t, sol)
        end
    end

    m, _ = hquadrature_v(integrand, 0, π, reltol=1e-3, abstol=1e-3)
    2m
end


function limit(f::Function, x0::Real, x1::Real, δ::Real=1e-2)
    next = last = f(x0)
    Δ = x1 - x0
    while true
        x1 = x0 + Δ
        next = f(x1)

        if (abs(next - last) / abs(mean([next, last]))) < δ
            break
        end

        last = next
        Δ /= 2
    end

    next
end

Σ_A(k::Real, s::Real, params::ModelParameters) = 2 * sum((abs2(v(k, params)) .* ((params.g) .^ 2)) ./ (s^2 .+ params.ε .^ 2))
Σ_A(k::Real, params::ModelParameters) = Σ_A(k, 0, params)

ρ_A(k::Real, params::ModelParameters) = 1 / 2π * 1 / (1 + Σ_A(k, params))

# function analytic_time_averaged_displacement(params::ModelParameters)
#     function integrand(k)
#         dϕ(k, params) * (1 / (2π) - ρ_A(k, params))
#     end

#     m, _ = hquadrature(integrand, 0, π, reltol=1e-5, abstol=1e-5)
#     2m
# end

function construct_residue_poly(k::Real, params::ModelParameters)
    function elementary_poly(ε)
        Polynomial([complex(BigFloat(0), BigFloat(ε)), BigFloat(1)])
    end


    full = prod(elementary_poly(ε) for ε in params.ε)
    weighted = Polynomial([0])
    gs = (abs2.(params.g) * abs2(v(k, params)))
    for (i, g) in enumerate(gs)
        weighted += g .* prod(elementary_poly(ε) for ε in params.ε[1:end.!=i])
    end
    return weighted + Polynomial([BigFloat(0), BigFloat(1)]) * full, full
end

function residues_poly(k::Real, params::ModelParameters)
    poly, full = construct_residue_poly(k, params)
    rts = roots(coeffs(poly), polish=true)

    # function refine_roots(r)
    #     find_zero(poly, r)
    # end

    # rts = map(refine_roots, rts)

    residuals = Vector{Complex{BigFloat}}(undef, length(rts))

    for (i, r) in enumerate(rts)
        residuals[i] = full(r) / (fromroots(rts[1:end.!=i])(r))
    end

    Vector{Float64}(imag.(rts)), Vector{Float64}(residuals)
end

function ρ_A(t::Real, frequencies::Vector{<:Number}, residues::Vector{<:Number})
    (sum(residues .* exp.(t * complex.(0, frequencies))) |> abs2) / 2π
end


ρ_A_mean_born(k::Real, T::Real, params::ModelParameters) = 1 / (2π * (1 + Σ_A(k, 1 / T, params)))
ρ_A_mean_born(k::Real, params::ModelParameters) = 1 / (2π * (1 + Σ_A(k, 0, params)))
ρ_A_mean(residues::Vector{<:Number}) = sum(abs2.(residues)) / 2π

function ρ_A_mean(T::Real, frequencies::Vector{<:Number}, residues::Vector{<:Number})
    if T == 0
        return 1
    end

    mean::Complex{BigFloat} = 0

    for i = 1:length(frequencies)
        for j = 1:length(frequencies)
            if j == i
                continue
            end

            ω_i, ω_j = frequencies[[i, j]]
            diff = complex(0, ω_j - ω_i)
            mean += conj(residues[i]) * residues[j] * (exp(diff * T) - 1) / diff
        end
    end

    mean /= T * 2π
    mean += ρ_A_mean(residues)
    real(mean)
end

function ρ_A_mean(T::Real, solution::WalkSolution)
    if T == 0
        return 1
    end

    mean::Complex{BigFloat} = 0

    N = size(solution.vectors)[2]
    residues = solution.vectors[1, :]
    for i = 1:N
        for j = 1:N
            if j == i
                continue
            end

            ω_i, ω_j = solution.energies[[i, j]]
            diff = complex(0, (ω_j - ω_i) * T)
            mean += conj(residues[i]) * residues[j] * (exp(diff) - 1) / diff
        end
    end

    mean += ρ_A_mean(solution)
    real(mean)
end

function ρ_A_mean(T1::Real, T2::Real, solution::WalkSolution)
    if T2 < T1
        return 0
    end

    mean::Complex{BigFloat} = 0
    T = T2 - T1
    N = size(solution.vectors)[2]
    residues = solution.vectors[1, :]

    for i = 1:N
        for j = 1:N
            if j == i
                continue
            end

            ω_i, ω_j = solution.energies[[i, j]]
            diff = complex(0, (ω_j - ω_i) * T)
            diff1 = complex(0, (ω_j - ω_i) * T1)
            diff2 = complex(0, (ω_j - ω_i) * T2)
            mean -= conj(residues[i]) * residues[j] * (exp(diff1) - exp(diff2)) / diff
        end
    end

    mean += ρ_A_mean(solution)
    real(mean)
end

ρ_A_mean(solution::WalkSolution) = sum(abs2.(solution.vectors[1, :]))


function analytic_time_averaged_displacement(T::Real, params::ModelParameters, integrand::Function)
    m, _ = hquadrature(k -> integrand(k, T, params), 0, π, reltol=1e-5, abstol=1e-5)
    2m
end
analytic_time_averaged_displacement(T::Real, params::ModelParameters) = analytic_time_averaged_displacement(T, params, integrand_diagonalization)


function analytic_time_averaged_displacement(T1::Real, T2::Real, params::ModelParameters)
    function integrand(k, T1, T2, params)
        sol = solution(k, params)
        dϕ(k, params) * (1 / 2π - ρ_A_mean(T1, T2, sol))
    end

    m, _ = hquadrature(k -> integrand(k, T1, T2, params), 0, π, reltol=1e-5, abstol=1e-5)
    2m
end


function integrand_born(k, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean_born(k, params))
end

function integrand_diagonalization(k, params)
    sol = solution(k, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean(sol))
end

function integrand_residue(k, params)
    _, r = residues_poly(k, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean(r))
end

function integrand_born(k, T, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean_born(k, T, params))
end

function integrand_diagonalization(k, T, params)
    sol = solution(k, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean(T, sol))
end

function integrand_residue(k, T, params)
    f, r = residues_poly(k, params)
    dϕ(k, params) * (1 / 2π - ρ_A_mean(T, f, r))
end


function analytic_time_averaged_displacement(params::ModelParameters, integrand::Function)
    m, _ = hquadrature(k -> integrand(k, params), 0, π, reltol=1e-5, abstol=1e-5)
    2m
end

analytic_time_averaged_displacement(params::ModelParameters) = analytic_time_averaged_displacement(params, integrand_diagonalization)


time_scale(params::ModelParameters) = 2π / minimum(abs.(params.ε))

function lamb_shift(params::ModelParameters, k::Real=0, relative::Bool=false)
    H = hamiltonian(k, params)
    ψ_A = if params.sw_approximation
        [1; zeros(num_bath_modes(params))]
    else
        [1; zeros(num_bath_modes(params) + 1)]
    end

    energies, ev = eigen(H)

    overlaps = (ψ_A'*ev.|>abs)[1, :]
    index = argmax(overlaps)

    isolated_energy = energies[index]
    energies = deleteat!(energies, index)

    shift = minimum(energies) - isolated_energy
    if relative
        shift /= minimum(abs.(energies[begin+1:end] .- energies[begin:end-1]))
    end

    shift
end

lamb_shift(params::ExtendedModelParameters, args...) = lamb_shift(ModelParameters(params), args...)

function optimal_bath_shift(params::ModelParameters, k::Real)
    sum(abs2.(params.g) ./ params.ε) * abs2(v(k, params)) / abs2(v(0, params))
end

optimal_bath_shift(params::ExtendedModelParameters, args...; kwargs...) = optimal_bath_shift(ModelParameters(params), args...; kwargs...)

function auto_shift_bath(params::ExtendedModelParameters, args...; kwargs...)
    @set params.ω_A = optimal_bath_shift(params, args...; kwargs...)
end

function recurrence_time(p::ModelParameters)::Real
    2π / minimum(p.ε[begin+1:end] - p.ε[begin:end-1])
end

recurrence_time(p::ExtendedModelParameters) = recurrence_time(p |> ModelParameters)

function minimal_N(ρ_A::Real, α::Real, J::Real, ω_c::Real)
    return ((ω_c^2 / J * (1 / sqrt(ρ_A) - 1) / (α + 1) + 1 / (1 - α)) / zeta(2 - α))^(1 / (1 - α))
end

minimal_N(ρ_A::Real, sd::OhmicSpectralDensity) = minimal_N(ρ_A, sd.α, sd.J, sd.ω_c)
minimal_N(ρ_A::Real, params::ExtendedModelParameters) = minimal_N(ρ_A, params.spectral_density)

decay_rate(params::ExtendedModelParameters, k::Real=π) = 2π * params.spectral_density(params.ω_A) * abs2(v(k, params))
end
