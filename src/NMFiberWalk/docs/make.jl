#push!(LOAD_PATH,"../src/")
using Documenter, NMFibreWalk, NMFibreWalk.BathDiscretizations

makedocs(
    sitename="Non Markovian Quantum Walk on Fibre Loops Support Code",
    modules = [NMFibreWalk, NMFibreWalk.BathDiscretizations],
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo="github.com/vale981/finite_bath_nm_walk"
)
