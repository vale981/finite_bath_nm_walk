var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"This is the API documentation for the code related to simulating a non-markovian quantum walk in fibre loops.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [NMFibreWalk, NMFibreWalk.BathDiscretizations]\nOrder   = [:module, :type, :function]\nPrivate = false","category":"page"},{"location":"#NMFibreWalk.NMFibreWalk","page":"Home","title":"NMFibreWalk.NMFibreWalk","text":"An implementation of the non-markovian quantum walk in k-space with a discrete bath\n\n\n\n\n\n","category":"module"},{"location":"#NMFibreWalk.Diagonalization","page":"Home","title":"NMFibreWalk.Diagonalization","text":"Diagonalization(H)\nDiagonalization(k, params)\n\nThe result of diagonalizing the system hamiltonian.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.ExtendedModelParameters","page":"Home","title":"NMFibreWalk.ExtendedModelParameters","text":"ExtendedModelParameters([keyword args])\n\nThe parameters required to simulate the model, agnostic of how they are arrived at.  Instead of specifying the bath spectrum and coupling directly, these values are computed later on.\n\nv::Float64: Intracell hopping.\nu::Float64: Ratio of inter to intracell hopping.\nspectral_density::OhmicSpectralDensity: The spectral density to use for the bath.\nN::Int64: The number of bath modes.\ndiscretization::BathDiscretization: The bath discretization method to use. See BathDiscretization.\nnormalize::Bool: Whether to normalize the spectra density. See discretize_bath.\n\nsw_approximation::Bool: Whether the system should simulated in the Schrieffer-Wolff approximation.\n\nω_A::Float64: The energy of the A site.\nη0::Float64: The damping rate of the small loop.\nη0_bath::Float64: The damping rate of the big loop.\nη_coup::Float64: The damping rate due to the transmission line attached to the small loop.\nδ::Float64: The hybridization amplitude.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.ModelParameters","page":"Home","title":"NMFibreWalk.ModelParameters","text":"The parameters required to simulate the model, agnostic of how they are arrived at.\n\nv::Float64: Intracell hopping.\nu::Float64: Ratio of inter to intracell hopping.\nε::Vector{Float64}: Bath energy levels.\ng::Vector{Float64}: Bath coupling strengths.\nsw_approximation::Bool: Whether the system should simulated in the Schrieffer-Wolff approximation.\nω_A::Float64: Energy of the A site\nη::Vector{Float64}: The damping rates.\nψ::Float64: The asymmetry phase for the g.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.WalkSolution","page":"Home","title":"NMFibreWalk.WalkSolution","text":"WalkSolution(k, params[, m_0])\n\nA structure holding the information for the dynamic solution of the quantum walk for a specific k and with optional initial position m_0. Callable with a time parameter.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.WalkSolution-Tuple{Real}","page":"Home","title":"NMFibreWalk.WalkSolution","text":"(WalkSolution)(t)\n\nThe solution state at time t.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.dϕ-Tuple{Real, Real}","page":"Home","title":"NMFibreWalk.dϕ","text":"dϕ(k, u)\n\n\ndϕ(k::Real, u::Real) -> Real\n\n\nThe derivative of winding phase of the hopping amplitude. The arguments are as in v.\n\ndϕ(k, u)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:115.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.eigenmatrix-Tuple{Diagonalization}","page":"Home","title":"NMFibreWalk.eigenmatrix","text":"eigenmatrix(sol)\n\n\nA matrix with the eigenvectors of the hamiltonian as columns.\n\neigenmatrix(sol)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:258.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.hamiltonian-Tuple{Real, ModelParameters}","page":"Home","title":"NMFibreWalk.hamiltonian","text":"hamiltonian(k, params)\n\n\nhamiltonian(k, params)\n\nReturns the model Hamiltonian at momentum k for the params.  The basis is A bath levels.\n\nhamiltonian(k, params)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:216.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.inverse_eigenmatrix-Tuple{Diagonalization}","page":"Home","title":"NMFibreWalk.inverse_eigenmatrix","text":"inverse_eigenmatrix(sol)\n\n\nThe inverse of the matrix produced by eigenmatrix.\n\ninverse_eigenmatrix(sol)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:261.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.num_bath_modes-Tuple{ModelParameters}","page":"Home","title":"NMFibreWalk.num_bath_modes","text":"num_bath_modes(p)\n\n\nReturns the number of bath states for the model.\n\nnum_bath_modes(p)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:94.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.v-Tuple{Real, Real, Real}","page":"Home","title":"NMFibreWalk.v","text":"v(k, v, u)\n\n\nv(k::Real, v::Real, u::Real) -> Complex\n\n\nThe complex k rependent hopping amplitude, where v is the intracell hopping and u is the ratio of inter to intracell hopping.\n\nv(k, v, u)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:103.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.Δ-Tuple{ExtendedModelParameters}","page":"Home","title":"NMFibreWalk.Δ","text":"Δ(p)\n\n\nThe damping asymmetry such that the damping of the big loop is η_B = η_0 + 2δΔ.\n\nΔ(p)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:174.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.Ω-Tuple{Diagonalization}","page":"Home","title":"NMFibreWalk.Ω","text":"Ω(sol)\n\n\nThe complex eigenvalues of the the hamiltionian.\n\nΩ(sol)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:249.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.η0_A-Tuple{ExtendedModelParameters}","page":"Home","title":"NMFibreWalk.η0_A","text":"η0_A(params)\n\n\nThe damping rate of the A site.\n\nη0_A(params)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:181.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.η0_bath-Tuple{ExtendedModelParameters}","page":"Home","title":"NMFibreWalk.η0_bath","text":"η0_bath(params)\n\n\nThe damping rate of the bath modes.\n\nη0_bath(params)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:184.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.λ-Tuple{Diagonalization}","page":"Home","title":"NMFibreWalk.λ","text":"λ(sol)\n\n\nThe damping rates of the hamiltonian.\n\nλ(sol)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:255.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.ω-Tuple{Diagonalization}","page":"Home","title":"NMFibreWalk.ω","text":"ω(sol)\n\n\nThe eigenenergies of the hamiltonian.\n\nω(sol)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:252.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.ϕ-Tuple","page":"Home","title":"NMFibreWalk.ϕ","text":"ϕ(args)\n\n\nThe winding phase of the hopping amplitude. The arguments are as in v.\n\nϕ(args)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:108.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.ℰ-Tuple{ModelParameters}","page":"Home","title":"NMFibreWalk.ℰ","text":"ℰ(p)\n\n\nℰ(p::ModelParameters)\n\nThe complex site-energies ϵ_n - i γ^0_n.\n\nℰ(p)\n\ndefined at /home/hiro/Documents/org/roam/code/julia_code_project_for_the_non_markovian_quantum_walk_in_fiber_loops/src/NMFiberWalk/src/NMFibreWalk.jl:90.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.BathDiscretizations","page":"Home","title":"NMFibreWalk.BathDiscretizations","text":"Functionality related to the discretization of the spectral density into energies and coupling strengths. See discretize_bath.\n\n\n\n\n\n","category":"module"},{"location":"#NMFibreWalk.BathDiscretizations.BathDiscretization","page":"Home","title":"NMFibreWalk.BathDiscretizations.BathDiscretization","text":"BathDiscretization\n\nAn abstract type that defines how the bath is being discretized.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.BathDiscretizations.ExponentialBathDiscretization","page":"Home","title":"NMFibreWalk.BathDiscretizations.ExponentialBathDiscretization","text":"ExponentialBathDiscretization()\n\nDiscretize the bath with an exponential density of states ρ_f = Δ^-1 u^-\fracωΔ.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.BathDiscretizations.LinearBathDiscretization","page":"Home","title":"NMFibreWalk.BathDiscretizations.LinearBathDiscretization","text":"LinearBathDiscretization(integral_method=true, simple_energies=false)\n\nDiscretize the bath with a constant density of states.  The field integral_method controls whether the coupling strengths are computed using the integral of the spectral density or an approximation.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.BathDiscretizations.OhmicSpectralDensity","page":"Home","title":"NMFibreWalk.BathDiscretizations.OhmicSpectralDensity","text":"OhmicSpectralDensity(ω_c, J, α)\n\nAn ohmic spectral density of the form J(ω) = J \fracα+1ω_c^α+1 ω_c^α. Calling an instance evaluates the spectral density.\n\n\n\n\n\n","category":"type"},{"location":"#NMFibreWalk.BathDiscretizations.discretization_name-Tuple{LinearBathDiscretization}","page":"Home","title":"NMFibreWalk.BathDiscretizations.discretization_name","text":"discretization_name(discretization)\n\nReturns the name of the discretization.\n\n\n\n\n\n","category":"method"},{"location":"#NMFibreWalk.BathDiscretizations.discretize_bath","page":"Home","title":"NMFibreWalk.BathDiscretizations.discretize_bath","text":"discretize_bath(scheme, J, N[, normalize=true])\n\nDiscretize the bath using scheme according to the spectral density J into N energies and coupling strengths. Returns the energies ϵ and coupling strengths g as vectors. The coupling strengths can be optionally normalized so that _j g_j^2 = 1.\n\n\n\n\n\n","category":"function"}]
}