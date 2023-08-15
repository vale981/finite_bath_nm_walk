"""
Functionality related to the discretization of the spectral density
into energies and coupling strengths. See [`discretize_bath`](@ref).
"""
module BathDiscretizations

export BathDiscretization,
    LinearBathDiscretization,
    ExponentialBathDiscretization,
    OhmicSpectralDensity,
    discretization_name,
    discretize_bath,
    OhmicSpectralDensity

using Parameters

"""
    BathDiscretization

An abstract type that defines how the bath is being discretized.
"""
abstract type BathDiscretization end


"""
    LinearBathDiscretization(integral_method=true, simple_energies=false)

Discretize the bath with a constant density of states.  The field
`integral_method` controls whether the coupling strengths are computed
using the integral of the spectral density or an approximation.
"""
@with_kw struct LinearBathDiscretization <: BathDiscretization
    integral_method::Bool = true
    simple_energies::Bool = false
end

"""
    ExponentialBathDiscretization()

Discretize the bath with an exponential density of states ``ρ_{f} =
Δ^{-1} \eu^{-\frac{ω}{Δ}}``.
"""
struct ExponentialBathDiscretization <: BathDiscretization end


"""
    discretization_name(discretization)

Returns the name of the `discretization`.
"""
discretization_name(::LinearBathDiscretization) = "linear"
discretization_name(::ExponentialBathDiscretization) = "exponential"


"""
    use_integral_method(discretization)

Whether to use integration to obtain the coupling strengths for `discretization`.
"""
use_integral_method(::ExponentialBathDiscretization) = true
use_integral_method(params::LinearBathDiscretization) = params.integral_method


"""
    OhmicSpectralDensity(ω_c, J, α)

An ohmic spectral density of the form ``J(ω) = J \frac{α+1}{ω_c^{α+1}}
ω_c^α``. Calling an instance evaluates the spectral density.
"""
struct OhmicSpectralDensity{T<:Real}
    ω_c::T
    J::T
    α::T
end

(J::OhmicSpectralDensity)(ε::Real) =
    if ε < J.ω_c
        J.J * ε^J.α * ((J.α + 1) / J.ω_c^(J.α + 1))
    else
        0
    end

"""
    OhmicSpectralDensityIntegral(ω_c, J, α)

The antiderivate of the [`OhmicSpectralDensity`](@ref). See its
documentation for details.  This struct is callable.
"""
struct OhmicSpectralDensityIntegral{T<:Real}
    ω_c::T
    J::T
    α::T
end

"""
    OhmicSpectralDensityIntegral(J::OhmicSpectralDensity)

Instantiate an `OhmicSpectralDensityIntegral` from an `OhmicSpectralDensity` `J`.
"""
OhmicSpectralDensityIntegral(J::OhmicSpectralDensity) =
    OhmicSpectralDensityIntegral(J.ω_c, J.J, J.α)

(J::OhmicSpectralDensityIntegral)(ε::Real) =
    if ε < J.ω_c
        J.J * ε^(J.α + 1) / J.ω_c^(J.α + 1)
    else
        J.J
    end

"""
    discretize_bath(scheme, J, N[, normalize=true])

Discretize the bath using `scheme` according to the spectral density
`J` into `N` energies and coupling strengths. Returns the energies
``ϵ`` and coupling strengths ``g`` as vectors. The coupling strengths
can be optionally normalized so that ``∑_j g_j^2 = 1``.
"""
function discretize_bath(
    scheme::BathDiscretization,
    J::OhmicSpectralDensity,
    N::Integer,
    normalize::Bool = true,
)
    xk, ε = find_nodes_and_energies(scheme, J, N)

    if J.J == 0
        return ε, zero(ε)
    end

    g = if use_integral_method(scheme)
        J_int = OhmicSpectralDensityIntegral(J)
        ((J_int.(xk[2:end]) - J_int.(xk[1:end-1])))
    else
        dx = xk[2:end] - xk[1:end-1]
        J.(ε) .* dx
    end

    if normalize
        g /= sum(abs.(g))
        g *= J.J
    end

    ε, sqrt.(g)
end


"""
    find_nodes_and_energies(discretization::BathDiscretization, J, N)

Find the interval boundaries ``x_k`` and the energies ``ϵ_k`` for the
given ``discretization`` method, spectral density ``J`` and number of
bath states ``N``.
"""
function find_nodes_and_energies(
    discretization_params::LinearBathDiscretization,
    J::OhmicSpectralDensity,
    N::Integer,
)
    Δ = J.ω_c
    xk = collect(LinRange(0, Δ, N + 1))
    ε = if discretization_params.simple_energies
        xk[2:end]
    else
        Δ * ((2 * collect(1:N) .- 1) / (2 * N))
    end

    xk, ε
end

function find_nodes_and_energies(
    ::ExponentialBathDiscretization,
    J::OhmicSpectralDensity,
    N::Integer,
)
    ω_c = -J.ω_c / log(1 / (2N))
    xk = -ω_c * log.(1 .- collect(0:N) / (N))
    ε = -ω_c * log.(1 .- (2 * collect(1:N) .- 1) / (2 * N))

    xk, ε
end
end
