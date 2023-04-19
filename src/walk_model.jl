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
