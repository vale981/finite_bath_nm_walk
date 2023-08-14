module NMFiberWalk
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
export a_weight
export non_a_weight
export mean_displacement
export integrand_born
export integrand_diagonalization
export integrand_residue
export time_scale
export num_bath_modes
export BathDiscretization
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
export analytic_time_averaged_displacement_continuum
export Diagonalization
export O_σγO_γσ
export O_σγO_γn
export O_nγO_γσ
export O_nγO_γm
export Ω_γ
export Ω_n
export ω_γ_guess
export Κ
export energies
export Transmission
export Δ
export ψ

using Parameters
import LinearAlgebra: diagm, eigen, eigvals, ⋅
import Cubature: hquadrature, hquadrature_v
import Statistics: mean
import SpecialFunctions: gamma, zeta
import Accessors: @set
import Optim

abstract type BathDiscretization end
@kwdef struct LinearBathDiscretization <: BathDiscretization
    integral_method::Bool = true
    simple_energies::Bool = false
end

struct ExponentialBathDiscretization <: BathDiscretization
end
discretization_name(::LinearBathDiscretization) = "linear"
discretization_name(::ExponentialBathDiscretization) = "exponential"
use_integral_method(::ExponentialBathDiscretization) = true
use_integral_method(params::LinearBathDiscretization) = params.integral_method



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

    """The damping rates."""
    η::Vector{<:Real} = [0, 0, 0]
    @assert length(η) == length(ε) + 1

    """The asymmetry phase for the g."""
    ψ::Real = 0
end

Ω_n(p::ModelParameters) = p.ε - 1im .* p.η[2:end]
Ω_n(p::ModelParameters, n::Integer) = p.ε[n] - 1im .* p.η[1 + n]

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

    """The B site energy detuning."""
    ω::Real = 0.1

    spectral_density::OhmicSpectralDensity = OhmicSpectralDensity(1, 0.01, 1)

    N::Integer = 10
    discretization::BathDiscretization = LinearBathDiscretization()

    normalize::Bool = true

    """Whether the system should simulated in the Schrieffer-Wolff
       approximation."""
    sw_approximation::Bool = false

    ω_A::Real = 0

    """The damping rate of the small loop."""
    η0::Real = 0

    """The damping rate of the big loop."""
    η0_bath::Real = 0

    """The damping rate due to the transmission line attached to the small loop."""
    η_coup::Real = 0

    """The hybridization amplitude."""
    δ::Real = 0
end

"""The damping asymmetry such that the damping of the big loop is ``η_B = η_0 + 2δΔ``."""
Δ(p::ExtendedModelParameters) = (p.η0_bath - p.η0) / (2*p.δ)
raw"""The asymmetry phase `ψ=\sin^{-1Δ1}`The asymmetry phase `ψ=\sin^{-1Δ}`."""
ψ(params::ExtendedModelParameters) = asin(Δ(params))

"""The damping rate of the A site."""
η0_σ(params::ExtendedModelParameters) = params.η0 + η0_bath(params)

"""The damping rate of the bath modes."""
η0_bath(params::ExtendedModelParameters) = params.η0_bath

function ModelParameters(p::ExtendedModelParameters)
    ε, g = discretize_bath(p)
    η = [η0_σ(p) + p.η_coup / 2; fill(η0_bath(p) + p.η_coup, p.N)]


    ModelParameters(p.v, p.u, p.ω, ε, g, p.sw_approximation, p.ω_A, η, ψ(p))
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
        [params.ω_A (g' .* exp(-1im * params.ψ))
            g H_bath] - 1im .* diagm(0 => params.η)
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


struct Diagonalization
    O::Matrix{<:Complex}
    O_inv::Matrix{<:Complex}
    ω::Vector{<:Real}
    λ::Vector{<:Real}
end

Ω_γ(sol::Diagonalization) = sol.ω - 1im .* sol.λ


function Diagonalization(H::Matrix{<:Complex})
    energies, O = eigen(H)
    Diagonalization(O, inv(O), real.(energies), -imag.(energies))
end


energies(params::ModelParameters, k::Real=0) = eigvals(hamiltonian(k, params))

Diagonalization(k::Real, p::ExtendedModelParameters) = Diagonalization(hamiltonian(k, p))
Diagonalization(p::ExtendedModelParameters) = Diagonalization(0, p)
Diagonalization(p::ModelParameters) = Diagonalization(hamiltonian(0, p))

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


function discretize_bath(scheme::BathDiscretization, J::OhmicSpectralDensity, N::Integer, normalize::Bool=true)
    xk, ε = find_nodes_and_energies(scheme, J, N)

    if J.J == 0
        return ε, zero(ε)
    end

    g = if use_integral_method(scheme)
        J_int = OhmicSpectralDensityIntegral(J)
        ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))
    else
        dx = xk[2:end] - xk[1:end - 1]
        J.(ε) .* dx
    end

    if normalize
        g /= sum(abs.(g))
        g *= J.J
    end

    ε, sqrt.(g)
end
discretize_bath(p::ExtendedModelParameters) = discretize_bath(p.discretization, OhmicSpectralDensity(p), p.N, p.normalize)

function find_nodes_and_energies(discretization_params::LinearBathDiscretization, J::OhmicSpectralDensity, N::Integer)
    Δ = J.ω_c
    xk = collect(LinRange(0, Δ, N + 1))
    ε = if discretization_params.simple_energies
        xk[2:end]
    else
        Δ * ((2 * collect(1:N) .- 1) / (2 * N))
    end

    xk, ε
end

function find_nodes_and_energies(::ExponentialBathDiscretization, J::OhmicSpectralDensity, N::Integer)
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
    target(λ, params) = -imag(Κ(0, λ, params)) * abs2(v(k, params)) / abs2(v(0, params))
    first_guess = target(params.η[1], params)

    if isnan(first_guess)
        return 0
    end

    if params.ψ == 0 && maximum(abs.(params.η)) == 0
        first_guess
    else
        ω_c = maximum(params.ε)
        function gap(ω)
            params = @set params.ω_A = ω
            ω_0 = real(energies(params, 0)[1])
            abs(ω_0/params.ε[end])
        end

        (Optim.optimize(gap, first_guess * .8, first_guess * 1.2, iterations=100) |> Optim.minimizer)
    end
end

optimal_bath_shift(params::ExtendedModelParameters, args...; kwargs...) = optimal_bath_shift(ModelParameters(params), args...; kwargs...)

function auto_shift_bath(params::ExtendedModelParameters, args...; kwargs...)
    @set params.ω_A = optimal_bath_shift(params, args...; kwargs...)
end

function auto_shift_bath(params::ModelParameters, args...; kwargs...)
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

decay_rate(params::ExtendedModelParameters, k::Real=0) = π * params.spectral_density.J /params.spectral_density.ω_c * abs2(v(k, params)) / abs2(v(0, params))

function ρ_A_continuum(k::Real, p::ExtendedModelParameters)
    α = p.spectral_density.α
    if α < 1
        0
    else
        v_normed = abs2(v(k, p))/abs2(v(0, p))
        1 / (1 + p.spectral_density.J * (α+1) * v_normed / (p.spectral_density.ω_c^(2) * (α-1)))^2 * 1/2π
    end
end

function analytic_time_averaged_displacement_continuum(p::ExtendedModelParameters)
    reduced_params = ModelParameters(p)
    m, _ = hquadrature(k -> dϕ(k, reduced_params) * (1 / 2π - ρ_A_continuum(k, p)), 0, π, reltol=1e-5, abstol=1e-5)
    2m
end


# TODO: maybe optimize
V_σn(p::ModelParameters) = conj.(p.g) .* exp(-1im * p.ψ)
V_σn(p::ModelParameters, n::Integer) = conj(p.g[n]) .* exp(-1im * p.ψ)
V_nσ(p::ModelParameters) = p.g
V_nσ(p::ModelParameters, n::Integer) = p.g[n]

"""The self-energy ``Κ(-i Ω_γ)``"""
function Κ(Ω::Complex, p::ModelParameters)
    sum(V_σn(p) .* V_nσ(p) ./ (1im * (Ω_n(p) .- Ω)))
end

Κ(ω::Real, λ::Real, p::ModelParameters) = Κ(ω - 1im * λ, p)

""""""
function ∂Κ(Ω::Complex, p::ModelParameters)
    sum(V_σn(p) .* V_nσ(p) ./ (Ω_n(p) .- Ω).^2)
end

O_σγO_γσ(ω::Complex, p::ModelParameters) = 1 / (1 + ∂Κ(ω, p))
O_σγO_γn(ω::Complex, p::ModelParameters, n::Integer) = V_σn(p, n) / ((1 + ∂Κ(ω, p)) * (ω - Ω_n(p,n)))
O_nγO_γσ(ω::Complex, p::ModelParameters, n::Integer) = V_nσ(p, n) / ((1 + ∂Κ(ω, p)) * (ω - Ω_n(p,n)))
O_nγO_γm(ω::Complex, p::ModelParameters, n::Integer, m::Integer) = (V_nσ(p, n) * V_σn(p, m)) / ((1 + ∂Κ(ω, p)) * (ω - Ω_n(p,n)) * (ω - Ω_n(p,m)))


function ω_γ_guess(m::Integer, p::ModelParameters)
    ω_m =     Ω_n(p, m)
    ω_σ = (p.ω_A - 1im * p.η[1])
    A =  sum((V_σn(p) .* V_nσ(p))[1:end .!= m] ./ (Ω_n(p) .- ω_m)[1:end .!= m])
    B =  sum((V_σn(p) .* V_nσ(p))[1:end .!= m] ./ ((Ω_n(p) .- ω_m)[1:end .!= m]).^2)

    f_1 = (A + ω_m - (p.ω_A - 1im * p.η[1])) / (2 * (B + 1))

    (f_1 + sqrt(f_1^2 + V_σn(p, m) .* V_nσ(p, m)/ (B+1)))

    ω_m -(A  + ω_m - ω_σ - sqrt(4(1+B) * V_σn(p, m) .* V_nσ(p, m) + (A + ω_m - ω_σ)^2)) / (2 * (1+B))

    #1im * (Κ(Ω_λ(TargetDiagonal(p))[2], p) - 1im * Ω_λ(TargetDiagonal(p))[2]) - (p.ω_A - 1im * p.η[1])
end

@with_kw struct Transmission
    peak_positions::Vector{<:Real}
    peak_amplitudes::Vector{<:Complex}
    peak_widths::Vector{<:Real}
    κ::Real = 1e-3
    @assert length(peak_widths) == length(peak_amplitudes) == length(peak_positions)
end

function (t::Transmission)(ω::Real)
    F = t.κ * sum(t.peak_amplitudes .* (1 ./ (1im * (t.peak_positions .- ω) .+ t.peak_widths)))

    (1 - 2real(F) + abs2(F))
end

function Transmission(Ω_B::Real, κ::Real, full_params::ExtendedModelParameters, n::Integer=0)
    params = ModelParameters(full_params)
    trafo = Diagonalization(params)


    ε = params.ε
    η = params.η
    ψ = params.ψ

    if n == 0
        N = num_bath_modes(params)

        num_peaks = (N+1)^2
        peak_amplitudes = zeros(Complex, num_peaks)
        peak_widths = zeros(Real, num_peaks)
        peak_positions = zeros(Real, num_peaks)

        peak_amplitudes[begin:N+1] .= (trafo.O[1,:] .* trafo.O_inv[:, 1]) .* exp(1im * ψ) / (2 * cos(ψ))
        peak_widths[begin:N+1] .= trafo.λ
        peak_positions[begin:N+1] .= (full_params.δ - full_params.ω_A) .+ trafo.ω

        for n in 1:1:N
            rng = ((N+1)*n + 1):((N+1)*(n+1))
            peak_amplitudes[rng]  .= (trafo.O[n + 1, :] .* trafo.O_inv[:, n + 1])
            peak_widths[rng] .= trafo.λ
            peak_positions[rng] .= n * Ω_B - ε[n] .+ trafo.ω
        end

        Transmission(peak_positions, peak_amplitudes, peak_widths, κ)

    elseif n > 0

        N = num_bath_modes(params)
        num_peaks = (N+1)

        peak_positions = [fill(n * Ω_B - ε[n], num_peaks) .+ trafo.ω; (full_params.δ - full_params.ω_A) .+ trafo.ω]
        peak_widths = [trafo.λ; trafo.λ]
        peak_amplitudes = [(trafo.O[1, :] .* trafo.O_inv[:, n + 1]) ./ sqrt(2); (trafo.O[n + 1, :] .* trafo.O_inv[:, 1]) .* exp(-1im * ψ) / (sqrt(2) * cos(ψ))]
        Transmission(peak_positions, peak_amplitudes, peak_widths, κ)
    end
end
end # module NMFiberWalk
