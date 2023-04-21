module WalkModel
"""An implementation of the non-markovian quantum walk in k-space with a discrete bath"""

export ModelParameters
export v
export hamiltonian
export solution
export Σ_A
export analytic_time_averaged_displacement
export OhmicSpectralDensity
export linear_energy_distribution
export exponential_energy_distribution
export a_weight
export non_a_weight
export mean_displacement
export time_averaged_displacement

using Parameters
import LinearAlgebra: diagm, eigen
import DifferentialEquations as de
import Cubature: hquadrature, hquadrature_v
import Statistics: mean

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
end

"""Returns the number of bath states for the model."""
num_bath_modes(p::ModelParameters) = length(p.g)

"""
    v(k, v, u)

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

(J::OhmicSpectralDensity)(ε::Real) = ε <= J.Δ ? J.J * ε^J.α : 0

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
    H_bath = diagm(0 => params.ε)

    if params.sw_approximation
        g = abs(v_complex) * params.g

        [0 g'
            g H_bath]
    else
        V = [0 conj(v_complex)
            v_complex 0]

        H_AB = [0 0; 0 params.ω]

        H_system_bath = [zeros(num_bath_modes(params))'
            (params.g)']

        [(V+H_AB) H_system_bath
            H_system_bath' H_bath]
    end
end



function solution(k::Real, params::ModelParameters, m_0::Integer=0)
    H = hamiltonian(k, params)
    energies, vectors = eigen(H)
    energies = real.(energies)


    ψ_0 = if params.sw_approximation
        [exp(-1im * k * m_0); zeros(num_bath_modes(params))]
    else
        [exp(-1im * k * m_0); 0; zeros(num_bath_modes(params))]
    end

    coefficients = vectors' * ψ_0

    WalkSolution((coefficients' .* vectors), energies, params)
end

"""A structure holding the information for the dynamic solution of the
   quantum walk for a specific ``k``. Callable with a time parameter."""
struct WalkSolution
    vectors::Matrix{<:Complex}
    energies::Vector{<:Real}
    params::ModelParameters
end

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
a_weight(t::Real, sol::WalkSolution)::Real = sol(t)[begin] |> abs2
non_a_weight(t::Real, sol::WalkSolution)::Real = (1 - abs2(sol(t)[1]) / (2π))


"""Return `N` energies distributed according to ``exp(-ε/ω_c)`` in the
   interval `(0, J.Δ)`."""
function exponential_energy_distribution(J::OhmicSpectralDensity, N::Integer, ε_0::Real =0)
    ω_c = -J.Δ / log(1 / (2N))
    xk = -ω_c * log.(1 .- collect(0:N) / (N))
    ε = -ω_c * log.(1 .- (2 * collect(1:N) .- 1) / (2 * N))

    if ε_0 > 0
        ε[1] = ε_0
    end

    J_int = integral(J)
    xk[end] = min(xk[end], J_int.Δ)
    g = ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))
    ε, sqrt.(g)
end


function linear_energy_distribution(J::OhmicSpectralDensity, N::Integer, ε_0::Real =0)
    xk = collect(LinRange(0, J.Δ, N + 1))
    ε = J.Δ * ((2 * collect(1:N) .- 1) / (2 * N))

    if ε_0 > 0
        ε[1] = ε_0
    end

    J_int = integral(J)
    xk[end] = min(xk[end], J_int.Δ)
    g = ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))

    ε, sqrt.(g)
end

function mean_displacement(t::Real, params::ModelParameters, m_0::Integer=0)
    function integrand(ks, v)
        Threads.@threads for i = 1:length(ks)
            k = ks[i]
            sol = solution(k, params, m_0)
            @inbounds v[i] = dϕ(k, params) * non_a_weight(t, sol)
        end
    end

    m, _ = hquadrature_v(integrand, 0, π, reltol=1e-2, abstol=1e-2)
    m / (π)
end


function time_averaged_displacement(t::Tuple{Real,Real}, params::ModelParameters, m_0::Integer=0)
    function integrand(t)
        mean_displacement(t, params, m_0)
    end

    m, _ = hquadrature(integrand, t[1], t[2], reltol=1e-2, abstol=1e-2)
    m / (t[2] - t[1])
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

Σ_A(k::Real, s::Real, params::ModelParameters) = 2 * sum((abs2(v(k, params)) .* (params.g) .^ 2) .* s ./ (s^2 .+ params.ε .^ 2)) / s
Σ_A(k::Real, params::ModelParameters, δ::Real=1e-8) = limit(t -> Σ_A(k, t, params), 0, 1e-3, δ)

ρ_A(k::Real, params::ModelParameters, δ::Real=1e-8) = 1/2π * 1/(1+Σ_A(k, params, δ))

function analytic_time_averaged_displacement(params::ModelParameters, δ::Real=1e-8)
    function integrand(k)
        dϕ(k, params) * (1/(2π) - ρ_A(k, params, δ))
    end

    m, _ = hquadrature(integrand, 0, π, reltol=1e-5, abstol=1e-5)
    2m
end

end
