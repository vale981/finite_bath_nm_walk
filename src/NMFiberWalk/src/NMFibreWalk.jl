"""An implementation of the non-markovian quantum walk in k-space with a discrete bath"""
module NMFibreWalk

###############################################################################
#                                   Exports                                   #
###############################################################################

# basic parameter types
export ModelParameters,
    ExtendedModelParameters, ℰ, num_bath_modes, v, ϕ, dϕ, Δ, ψ, η0_A, η0_bath, V_An, V_nA

# dynamics
export hamiltonian
export Diagonalization, Ω, ω, λ, eigenmatrix, inverse_eigenmatrix
export WalkSolution, ρ_A, ρ_A_mean, ρ_A_continuum, time_averaged_displacement, time_averaged_displacement_continuum

# Transmission
export Transmission, AnalyticPeakAmplitudes

# misc
export minimal_N, recurrence_time, decay_rate

###############################################################################
#                                   Imports                                   #
###############################################################################

using Parameters
import LinearAlgebra: diagm, eigen, eigvals, ⋅
import Cubature: hquadrature, hquadrature_v
import Statistics: mean
import SpecialFunctions: gamma, zeta
import Accessors: @set
import Optim
using DocStringExtensions

include("NMFibreWalk/BathDiscretizations.jl")
using .BathDiscretizations

@template (FUNCTIONS, METHODS, MACROS) = """
                                         $(SIGNATURES)
                                         $(DOCSTRING)
                                         $(METHODLIST)
                                         """

@template STRUCTS = """
                    $(TYPEDEF)
                    $(DOCSTRING)
                    $(TYPEDFIELDS)
                    """


###############################################################################
#                               Basic Parameters                              #
###############################################################################

"""The parameters required to simulate the model, agnostic of how they
are arrived at.

$(TYPEDFIELDS)
"""
@kwdef struct ModelParameters
    """Intracell hopping."""
    v::Float64 = 1

    """Ratio of inter to intracell hopping."""
    u::Float64 = 0.5

    """Bath energy levels."""
    ε::Vector{Float64} = [1.0, 2.0]

    """Bath coupling strengths."""
    g::Vector{Float64} = [1.0, 2.0]
    #@assert length(g) == length(ε)

    """Whether the system should simulated in the Schrieffer-Wolff
       approximation."""
    sw_approximation::Bool = false

    """Energy of the A site"""
    ω_A::Float64 = 0

    """The damping rates."""
    η::Vector{Float64} = [0, 0, 0]
    #@assert length(η) == length(ε) + 1

    """The asymmetry phase for the g."""
    ψ::Float64 = 0
end


"""
    ℰ(p::ModelParameters)

The complex site-energies ``ϵ_n - i γ^0_n``.
"""
ℰ(p::ModelParameters) = p.ε - 1im .* p.η[2:end]
ℰ(p::ModelParameters, n::Integer) = p.ε[n] - 1im .* p.η[1+n]

"""Returns the number of bath states for the model."""
num_bath_modes(p::ModelParameters) = length(p.g)

"""
    $(TYPEDSIGNATURES)

The complex ``k`` rependent hopping amplitude, where `v` is the
intracell hopping and `u` is the ratio of inter to intracell
hopping.
"""
v(k::Real, v::Real, u::Real)::Complex = v + u * v * exp(complex(0, k))
v(k::Real, params::ModelParameters)::Complex = v(k, params.v, params.u)

"""The winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
ϕ(args...)::Real = angle(v(args...))

"""
    $(TYPEDSIGNATURES)
The derivative of winding phase of the hopping amplitude.
The arguments are as in [`v`](@ref).
"""
dϕ(k::Real, u::Real)::Real = u * (u + cos(k)) / (u^2 + 1 + 2 * u * cos(k))
dϕ(k::Real, params::ModelParameters)::Real = dϕ(k, params.u)

"""
    ExtendedModelParameters([keyword args])

The parameters required to simulate the model, agnostic of how they
are arrived at.  Instead of specifying the bath spectrum and coupling
directly, these values are computed later on.

$(TYPEDFIELDS)
"""
@kwdef struct ExtendedModelParameters
    """Intracell hopping."""
    v::Float64 = 1

    """Ratio of inter to intracell hopping."""
    u::Float64 = 0.5

    """The spectral density to use for the bath."""
    spectral_density::OhmicSpectralDensity = OhmicSpectralDensity(1, 0.01, 1)

    """The number of bath modes."""
    N::Int64 = 10 # no point in modesty

    """The bath discretization method to use. See [`BathDiscretization`](@ref)."""
    discretization::BathDiscretization = LinearBathDiscretization()

    """
    Whether to normalize the spectra density.
    See [`discretize_bath`](@ref).
    """
    normalize::Bool = true

    """
    Whether the system should simulated in the Schrieffer-Wolff
    approximation.
    """
    sw_approximation::Bool = false

    """The energy of the ``A`` site."""
    ω_A::Float64 = 0

    """The damping rate of the small loop."""
    η0::Float64 = 0

    """The damping rate of the big loop."""
    η0_bath::Float64 = 0

    """The damping rate due to the transmission line attached to the small loop."""
    η_coup::Float64 = 0

    """The hybridization amplitude."""
    δ::Float64 = 0
end

v(k::Real, params::ExtendedModelParameters)::Complex = v(k, params.v, params.u)

"""The damping asymmetry such that the damping of the big loop is ``η_B = η_0 + 2δΔ``."""
Δ(p::ExtendedModelParameters) = (p.η0_bath - p.η0) / (2 * p.δ)

raw"""The asymmetry phase `ψ=\sin^{-1Δ1}`The asymmetry phase `ψ=\sin^{-1Δ}`."""
ψ(params::ExtendedModelParameters) = asin(Δ(params))
ψ(params::ModelParameters) = params.ψ

"""The damping rate of the A site."""
η0_A(params::ExtendedModelParameters) = params.η0 + η0_bath(params)

"""The damping rate of the bath modes."""
η0_bath(params::ExtendedModelParameters) = params.η0_bath

## conversions
function ModelParameters(p::ExtendedModelParameters)
    ε, g = discretize_bath(p)
    η = [η0_A(p) + p.η_coup / 2; fill(η0_bath(p) + p.η_coup, p.N)]


    ModelParameters(p.v, p.u, p.ω, ε, g, p.sw_approximation, p.ω_A, η, ψ(p))
end

function convert(::Type{ModelParameters}, p::ExtendedModelParameters)
    ModelParameters(p)
end

BathDiscretizations.discretize_bath(p::ExtendedModelParameters) =
    discretize_bath(p.discretization, OhmicSpectralDensity(p), p.N, p.normalize)
BathDiscretizations.OhmicSpectralDensity(params::ExtendedModelParameters) =
    params.spectral_density
convert(::Type{OhmicSpectralDensity}, params::ExtendedModelParameters) =
    OhmicSpectralDensity(params)

###############################################################################
#                                   Dynamics                                  #
###############################################################################

"""
  hamiltonian(k, params)

Returns the model Hamiltonian at momentum ``k`` for the `params`.  The
basis is ``A, bath levels``.
"""
function hamiltonian(k::Real, params::ModelParameters)::Matrix{<:Number}
    v_complex = v(k, params)
    H_bath = diagm(0 => params.ε)

    g = v_complex .* params.g / abs(v(0, params))
    [
        params.ω_A (g'.*exp(-1im * params.ψ))
        g H_bath
    ] - 1im .* diagm(0 => params.η)
end


"""
    Diagonalization(H)
    Diagonalization(k, params)

The result of diagonalizing the system hamiltonian.
"""
struct Diagonalization
    """The diagonalizing transformation."""
    O::Matrix{ComplexF64}

    """The inverse diagonalizing transformation."""
    O_inv::Matrix{ComplexF64}

    """The eigenenergies."""
    ω::Vector{Float64}

    """The damping rates."""
    λ::Vector{Float64}
end

"""The complex eigenvalues of the the hamiltionian."""
Ω(sol::Diagonalization) = sol.ω - 1im .* sol.λ

"""The eigenenergies of the hamiltonian."""
ω(sol::Diagonalization) = sol.ω

"""The damping rates of the hamiltonian."""
λ(sol::Diagonalization) = sol.λ

"""A matrix with the eigenvectors of the hamiltonian as columns."""
eigenmatrix(sol::Diagonalization) = sol.O

"""The inverse of the matrix produced by [`eigenmatrix`](@ref)."""
inverse_eigenmatrix(sol::Diagonalization) = sol.O_inv


function Diagonalization(H::Matrix{<:Complex})
    energies, O = eigen(H)
    Diagonalization(O, inv(O), real.(energies), -imag.(energies))
end

function Diagonalization(k::Real, params::ModelParameters)
    Diagonalization(hamiltonian(k, params))
end


Diagonalization(k::Real, p::ExtendedModelParameters) = Diagonalization(hamiltonian(k, p))
Diagonalization(p::ExtendedModelParameters) = Diagonalization(0, p)
Diagonalization(p::ModelParameters) = Diagonalization(hamiltonian(0, p))



"""
    WalkSolution(k, params[, m_0])

A structure holding the information for the dynamic solution of the
quantum walk for a specific ``k`` and with optional initial position
`m_0`. Callable with a time parameter.
"""
struct WalkSolution
    vectors::Matrix{<:Complex}
    energies::Vector{<:Complex}
    params::ModelParameters
end

function WalkSolution(k::Real, params::ModelParameters, m_0::Integer=0)
    diag = Diagonalization(k, params)
    ψ_0 = [exp(-1im * k * m_0); 0; zeros(num_bath_modes(params))]

    coefficients =
        eigenmatrix(diag) .* inverse_eigenmatrix(diag)[:, 1] .* exp(-1im * k * m_0) ./
        sqrt(2π)


    WalkSolution(coefficients, Ω(diag), params)
end

WalkSolution(k::Real, params::ExtendedModelParameters, m_0::Integer=0) =
    WalkSolution(k, params |> ModelParameters, m_0)

"""
   (WalkSolution)(t)

The solution state at time ``t``."""
function (sol::WalkSolution)(t::Real)
    @inbounds sol.vectors * (exp.(-1im * sol.energies * t))
end


"""The ``\rho_{\bar{A}}(k, t)`` at time ``t`` for the specific solution `sol`."""
ρ_A(t::Real, sol::WalkSolution)::Real =
    sol.vectors[1, :] ⋅ (exp.(-1im * sol.energies * t)) |> abs2
function ρ_A(p::ExtendedModelParameters, k::Real=0)
    sol = WalkSolution(k, p |> ModelParameters)
    return t -> ρ_A(t, sol)
end

"""The probability density (in ``k``) to be off the chain."""
non_a_weight(t::Real, sol::WalkSolution)::Real = (1 / (2π) - abs2(sol(t)[1]))


raw"""
    mean_displacement(t, params[, m_0])

The mean displacement ``\langle m(t)\rangle`` at time `t`. Optionally
the initial position `m_0` can be specified.
"""
function mean_displacement(t::Real, params::ModelParameters, m_0::Integer=0)
    function integrand(ks, v)
        Threads.@threads for i = 1:length(ks)
            k = ks[i]
            sol = WalkSolution(k, params, m_0)
            @inbounds v[i] = dϕ(k, params) * non_a_weight(t, sol)
        end
    end

    m, _ = hquadrature_v(integrand, 0, π, reltol=1e-3, abstol=1e-3)
    2m
end

"""
    ρ_A_mean(T, solution)

The value of `ρ_A` time averaged over `[0, T]`.
"""
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

"""
    ρ_A_mean(T1, T2, solution)

The value of `ρ_A` time averaged over `[T1, T2]`.
"""
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

"""
    ρ_A_mean(solution)

The value of `ρ_A` time averaged over `[0, ∞]`.
"""
ρ_A_mean(solution::WalkSolution) = sum(abs2.(solution.vectors[1, :]))


raw"""
   time_averaged_displacement(params::ModelParameters)
   time_averaged_displacement(params::ModelParameters, T)
   time_averaged_displacement(params::ModelParameters, T1, T2)

The time averaged mean displacement `\langle m\rangle`, average over a
time analogous to the behavior of [`ρ_A_mean`](@ref).
"""
function time_averaged_displacement(params::ModelParameters, times...)
    function integrand(k)
        sol = WalkSolution(k, params)
        dϕ(k, params) * (1 / 2π - ρ_A_mean(sol, times...))
    end
    m, _ = hquadrature(integrand, 0, π, reltol=1e-5, abstol=1e-5)
    2m
end

raw"""
    optimal_bath_shift(params[, k=0])

The value of ``ϵ_A`` that compensates the lamb shift. If there is no
non-Hermiticity, an exact formula is utilized. Otherwise the result is
refined with numerics.
"""
function optimal_bath_shift(params::ModelParameters, k::Real=0)
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
            abs(ω_0 / params.ε[end])
        end

        (
            Optim.optimize(gap, first_guess * 0.8, first_guess * 1.2, iterations=100) |>
            Optim.minimizer
        )
    end
end

optimal_bath_shift(params::ExtendedModelParameters, args...; kwargs...) =
    optimal_bath_shift(ModelParameters(params), args...; kwargs...)


raw"""
    auto_shift_bath(params, ...)

Returns `params` with the value of `ω_A` set to the result of
[`optimal_bath_shift`](@ref), where all arguments are passed on to
this function.
"""
function auto_shift_bath(params::ExtendedModelParameters, args...; kwargs...)
    @set params.ω_A = optimal_bath_shift(params, args...; kwargs...)
end

function auto_shift_bath(params::ModelParameters, args...; kwargs...)
    @set params.ω_A = optimal_bath_shift(params, args...; kwargs...)
end


"""
    recurrence_time(params)

Returns the estimated recurrence time for `ρ_A`.
"""
function recurrence_time(p::ModelParameters)::Real
    2π / minimum(p.ε[begin+1:end] - p.ε[begin:end-1])
end

recurrence_time(p::ExtendedModelParameters) = recurrence_time(p |> ModelParameters)

"""
    minimal_N(ρ_A, α, J, ω_c)
    minimal_N(ρ_A, sd::OhmicSpectralDensity)
    minimal_N(ρ_A, params::ExtendedModelParameters)

The minimal number of bath modes needed to achieve a certain
asymptotic value of `ρ_A`.
"""
function minimal_N(ρ_A::Real, α::Real, J::Real, ω_c::Real)
    return (
        (ω_c^2 / J * (1 / sqrt(ρ_A) - 1) / (α + 1) + 1 / (1 - α)) / zeta(2 - α)
    )^(1 / (1 - α))
end

minimal_N(ρ_A::Real, sd::OhmicSpectralDensity) = minimal_N(ρ_A, sd.α, sd.J, sd.ω_c)
minimal_N(ρ_A::Real, params::ExtendedModelParameters) =
    minimal_N(ρ_A, params.spectral_density)

"""
    decay_rate(params[, k::Real=0])

The approximate decay rate of `ρ_A`.
"""
decay_rate(params::ExtendedModelParameters, k::Real=0) =
    π * params.spectral_density.J / params.spectral_density.ω_c * abs2(v(k, params)) /
    abs2(v(0, params))


"""
    ρ_A_continuum(k, p::ExtendedModelParameters)

A continuum estimate for the asymptotic value of `ρ_A`.
"""
function ρ_A_continuum(k::Real, p::ExtendedModelParameters)
    α = p.spectral_density.α
    if α < 1
        0
    else
        v_normed = abs2(v(k, p)) / abs2(v(0, p))
        1 /
        (
            1 +
            p.spectral_density.J * (α + 1) * v_normed /
            (p.spectral_density.ω_c^(2) * (α - 1))
        )^2 * 1 / 2π
    end
end

"""
    ρ_A_continuum(k, p::ExtendedModelParameters)

A continuum estimate for the long-time averaged mean displacement.
"""
function time_averaged_displacement_continuum(p::ExtendedModelParameters)
    reduced_params = ModelParameters(p)
    m, _ = hquadrature(
        k -> dϕ(k, reduced_params) * (1 / 2π - ρ_A_continuum(k, p)),
        0,
        π,
        reltol=1e-5,
        abstol=1e-5,
    )
    2m
end


"""
The matrix element ``V^0_{An}`` either as a vector for all ``n`` or
for a specific `n`. Here ``n`` should be positive.
"""
V_An(p::ModelParameters) = conj.(p.g) .* exp(-1im * p.ψ)
V_An(p::ModelParameters, n::Integer) = conj(p.g[n]) .* exp(-1im * p.ψ)

"""
The matrix element ``V^0_{nA}`` either as a vector for all ``n`` or
for a specific `n`. Here ``n`` should be positive.
"""
V_nA(p::ModelParameters) = p.g
V_nA(p::ModelParameters, n::Integer) = p.g[n]

"""
The self-energy ``Κ(-i Ω_γ)``
"""
function Κ(Ω::Complex, p::ModelParameters)
    sum(V_An(p) .* V_nA(p) ./ (1im * (Ω(p) .- Ω)))
end

Κ(ω::Real, λ::Real, p::ModelParameters) = Κ(ω - 1im * λ, p)

"""
The derivative of the self-energy ``∂Κ(-i Ω_γ)``
"""
function ∂Κ(Ω::Complex, p::ModelParameters)
    sum(V_An(p) .* V_nA(p) ./ (Ω(p) .- Ω) .^ 2)
end


"""
The peak amplitude ``O_{mγ} O^{-1}_{γn}`` as a function of the model
parameters `p`.
"""
AnalyticPeakAmplitudes(m::Integer, n::Integer, p) =
    if m == 0 && n == 0
        1 / (1 + ∂Κ(ω, p))
    elseif m == 0 && n > 0
        V_An(p, n) / ((1 + ∂Κ(ω, p)) * (ω - ℰ(p, n)))
    elseif m > 0 && n == 0
        V_nA(p, n) / ((1 + ∂Κ(ω, p)) * (ω - ℰ(p, n)))
    elseif m > 0 && n > 0
        (V_nA(p, n) * V_An(p, m)) / ((1 + ∂Κ(ω, p)) * (ω - ℰ(p, n)) * (ω - ℰ(p, m)))
    end

raw"""
An approximation for the `m`th eigenvalue of the target Hamiltonian
``\mathcal{H}``.
"""
function Ωγ_guess(m::Integer, p::ModelParameters)
    ω_m = ℰ(p, m)
    ω_σ = (p.ω_A - 1im * p.η[1])
    A = sum((V_An(p).*V_nA(p))[1:end.!=m] ./ (ℰ(p).-ω_m)[1:end.!=m])
    B = sum((V_An(p).*V_nA(p))[1:end.!=m] ./ ((ℰ(p).-ω_m)[1:end.!=m]) .^ 2)

    f_1 = (A + ω_m - (p.ω_A - 1im * p.η[1])) / (2 * (B + 1))

    (f_1 + sqrt(f_1^2 + V_An(p, m) .* V_nA(p, m) / (B + 1)))

    ω_m -
    (A + ω_m - ω_σ - sqrt(4(1 + B) * V_An(p, m) .* V_nA(p, m) + (A + ω_m - ω_σ)^2)) /
    (2 * (1 + B))

    #1im * (Κ(Ω_λ(TargetDiagonal(p))[2], p) - 1im * Ω_λ(TargetDiagonal(p))[2]) - (p.ω_A - 1im * p.η[1])
end




"""A container to hold the information to construct the transmission
   amplitude as a function of the laser frequency."""
@kwdef struct Transmission
    peak_positions::Vector{Float64}
    peak_amplitudes::Vector{ComplexF64}
    peak_widths::Vector{Float64}
    κ::Float64 = 1e-3
    harmonic::Int64 = 0
    #@assert length(peak_widths) == length(peak_amplitudes) == length(peak_positions)
end

"""
The transmission frequency as a function of laser frequency ``ω``.  If
the harmonic of the transmission is zero, the stationary transmission
intensity is returned. For higher harmonics the absolute value of the
Fourier component is returned.
"""
function (t::Transmission)(ω::Real)
    F = t.κ *
        sum(t.peak_amplitudes .* (1 ./ (1im * (t.peak_positions .- ω) .+ t.peak_widths)))

    if t.harmonic == 0
        (1 - 2real(F) + abs2(F))
    else
        abs(F)
    end
end

"""
Construct the transmission amplitude for the fourier component with
frequency ``ω^0_m-ω^0_n`` from the free spectral range of the big loop
``Ω_B``, the model parameters `full_params`.
"""
function Transmission(
    Ω_B::Real,
    full_params::ExtendedModelParameters,
    n::Integer=0,
)
    params = ModelParameters(full_params)
    trafo = Diagonalization(params)


    ε = params.ε
    ψ = params.ψ
    κ = full_params.η_coup

    if n == 0
        N = num_bath_modes(params)

        num_peaks = (N + 1)^2
        peak_amplitudes = zeros(Complex, num_peaks)
        peak_widths = zeros(Real, num_peaks)
        peak_positions = zeros(Real, num_peaks)

        peak_amplitudes[begin:N+1] .=
            (trafo.O[1, :] .* trafo.O_inv[:, 1]) .* exp(1im * ψ) / (2 * cos(ψ))
        peak_widths[begin:N+1] .= trafo.λ
        peak_positions[begin:N+1] .= (full_params.δ - full_params.ω_A) .+ trafo.ω

        for n = 1:1:N
            rng = ((N+1)*n+1):((N+1)*(n+1))
            peak_amplitudes[rng] .= (trafo.O[n+1, :] .* trafo.O_inv[:, n+1])
            peak_widths[rng] .= trafo.λ
            peak_positions[rng] .= n * Ω_B - ε[n] .+ trafo.ω
        end

        Transmission(peak_positions, peak_amplitudes, peak_widths, κ)

    elseif n > 0

        N = num_bath_modes(params)
        num_peaks = (N + 1)

        peak_positions = [
            fill(n * Ω_B - ε[n], num_peaks) .+ trafo.ω
            (full_params.δ - full_params.ω_A) .+ trafo.ω
        ]
        peak_widths = [trafo.λ; trafo.λ]
        peak_amplitudes = [
            (trafo.O[1, :] .* trafo.O_inv[:, n+1]) ./ sqrt(2)
            (trafo.O[n+1, :] .* trafo.O_inv[:, 1]) .* exp(-1im * ψ) / (sqrt(2) * cos(ψ))
        ]
        Transmission(peak_positions, peak_amplitudes, peak_widths, κ)
    end
end

end # module NMFiberWalk
