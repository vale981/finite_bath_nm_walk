"""An implementation of the non-markovian quantum walk in k-space with a discrete bath"""

using Parameters
import LinearAlgebra: diagm, eigen
import DifferentialEquations as de
import Cubature: hquadrature_v

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
end

"""Returns the number of bath states for the model."""
num_bath_modes(p::ModelParameters) = length(p.g)
num_bath_modes(H::Matrix{<:Number}) = size(H, 1) - 2

"""  v(k, v, u)

The complex ``k`` rependent hopping amplitude, where `v` is the
intracell hopping and `u` is the ratio of inter to intracell
hopping."""
v(k::Real, v::Real, u::Real)::Complex = v + u * v * exp(complex(0, k))
v(k::Real, params::ModelParameters)::Complex = v(k, params.v, params.u)

struct OhmicSpectralDensity
    Δ::Real
    J::Real
    α::Real
end

(J::OhmicSpectralDensity)(ε::Real) = ε <= J.Δ ? J.J * ε^J.α * (J.α + 1)/J.Δ^(J.α+1) : 0

integral(J::OhmicSpectralDensity) = OhmicSpectralDensity(J.Δ, J.J / (J.α + 1), J.α + 1)


"""The winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
ϕ(args...)::Real = angle(v(args...))

"""The derivative of winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
dϕ(k::Real, v::Real, u::Real)::Real = u * (u + cos(k)) / (u^2 + 1 + 2 * u * cos(k))
dϕ(k::Real, params::ModelParameters)::Real = dϕ(k, params.v, params.u)

"""
  hamiltonian(k, params)

Returns the model Hamiltonian at momentum ``k`` for the `params`.  The
basis is ``A, B, bath levels``.
"""
function hamiltonian(k::Real, params::ModelParameters)::Matrix{<:Number}
    v_complex = v(k, params)

    V = [0 conj(v_complex)
        v_complex 0]

    H_AB = [0 0; 0 params.ω]

    H_bath = diagm(0 => params.ε)

    H_system_bath = [zeros(num_bath_modes(params))'
        (params.g)']

    H = [(V+H_AB) H_system_bath
        H_system_bath' H_bath]

    H
end



solution(k::Real, params::ModelParameters, m_0::Integer=0) = solution(k, hamiltonian(k, params), m_0)
function solution(k::Real, H::Matrix{<:Complex}, m_0::Integer=0)
    energies, vectors = eigen(H)
    energies = real.(energies)

    ψ_0 = [exp(-1im * k * m_0); 0; zeros(num_bath_modes(H))]
    coefficients = vectors' * ψ_0

    WalkSolution((coefficients' .* vectors), energies)
end

"""A structure holding the information for the dynamic solution of the
   quantum walk for a specific ``k``. Callable with a time parameter."""
struct WalkSolution
    vectors::Matrix{<:Complex}
    energies::Vector{<:Real}
end

"""
   (WalkSolution)(t)

The solution at time ``t``."""
function (sol::WalkSolution)(t::Real)
    @inbounds sol.vectors * (exp.(complex.(0,-sol.energies * t)))
end

"""The ``\rho_{\bar{A}}(k, t)`` at time ``t`` for the specific solution `sol`."""
non_a_weight(t::Real, sol::WalkSolution)::Real = sol(t)[2:end] .|> abs2 |> sum
a_weight(t::Real, sol::WalkSolution)::Real = sol(t)[begin] |> abs2
#non_a_weight(t::Real, sol::WalkSolution)::Real = 1 - abs2(sol(t)[1]) / (2π)


"""Return `N` energies distributed according to ``exp(-ε/ω_c)`` in the
   interval `(0, J.Δ)`."""
function exponential_energy_distribution(J::OhmicSpectralDensity, N::Integer)
    ω_c = -J.Δ / log(1 / (2N))
    xk = -ω_c * log.(1 .- collect(0:N) / (N))
    ε = -ω_c * log.(1 .- (2 * collect(1:N) .- 1) / (2 * N))

    J_int = integral(J)
    xk[end] = min(xk[end], J_int.Δ)
    g = 1 / π * ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))
    ε, sqrt.(g)
end

function linear_energy_distribution(J::OhmicSpectralDensity, N::Integer)
    xk = collect(LinRange(0, J.Δ, N + 1))
    ε = J.Δ * ((2 * collect(1:N) .- 1) / (2 * N))

    J_int = integral(J)
    xk[end] = min(xk[end], J_int.Δ)
    g = 1 / π * ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))
    ε, sqrt.(g)
end



function mean_displacement(t::Real, params::ModelParameters, N::Integer, m_0::Integer=0)
    function integrand(ks, v)
        Threads.@threads for i = 1:length(ks)
            k = ks[i]
            sol = solution(k, params, m_0)
            @inbounds v[i] = dϕ(k, params) * non_a_weight(t, sol)
        end
    end

    m, _ = hquadrature_v(integrand, -π, π, reltol=1e-2, abstol=1e-2)
    m / (2π)
end
