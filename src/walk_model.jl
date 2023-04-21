"""An implementation of the non-markovian quantum walk in k-space with a discrete bath"""



"""The parameters required to simulate the model, agnostic of how they
are arrived at."""
struct ModelParameters
    """Intracell hopping."""
    v::Real = 1

    """Ratio of inter to intracell hopping."""
    u::Real = 0.5

    """Bath energy levels."""
    ε::Vector{Real} = []

    """Bath coupling strengths."""
    ω::Vector{Real} = []
end


"""  v(k, v, u)

The complex ``k`` rependent hopping amplitude, where `v` is the
intracell hopping and `u` is the ratio of inter to intracell
hopping."""
v(k::Real, v::Real, u::Real)::Real = v + u * v * exp(k)
v(k::Real, params::ModelParameters)::Real = v(k, params.v, params.u)

"""The winding phase of the hopping amplitude.
   The arguments are as in [`v`](@ref)."""
Φ(args...)::Real = arg(v(args...))

"""
  hamiltonian(k, params)

Returns the model Hamiltonian at momentum ``k`` for the `params`.  The
basis is ``A, B, bath levels``.
"""
function hamiltonian(k::Real, params::ModelParameters)
    v_complex = v(k, params)

    V = [0 conj(v_complex)
        v_complex 0]
end
