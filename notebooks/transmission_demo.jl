### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 9d844d42-ca07-403b-864b-8c70be71b4f2
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
end


# ╔═╡ 2e29b1b6-3584-11ee-1981-0b7192ce1847
begin
	using DrWatson
	@quickactivate "discrete_walk"
	using Revise
	using LinearAlgebra
	using Accessors
	using Statistics
	using SpecialFunctions
	using LaTeXStrings
	#using CairoMakie
	using Latexify
	#CairoMakie.activate!(type = "svg")
	using PlutoUI
end

# ╔═╡ 40ba9793-d02f-4599-929a-4352f3712c82
begin
		include(srcdir("WalkModel.jl"))
		using .WalkModel
	    include(srcdir("Utilities.jl"))
		using .Utilities
	     
end

# ╔═╡ 3fc99094-b156-4d6c-b176-9ea684745ece
@bind n Slider(0:1:5)

# ╔═╡ 59923ada-370f-4a9c-9920-08a792bf70ba
@bind j Slider(.00:.01:2, default=1)

# ╔═╡ 2d11dd54-2314-4c47-ade9-7636384f1181
@bind α Slider(0:.1:2)

# ╔═╡ c61dec24-e44f-4b84-956a-aa5f4b11e119
α,j,n

# ╔═╡ 23cb9bd6-6bbe-4b41-834a-02d1de30d13d
begin
	η_coup = .001
	@bind test html"alpha"
	(full_prototype, prototype, spectral_density) = let
	    v = 1
	    u = 1
	    ω_c = .1
	    N = 5
	    J = (.2  * ω_c)^2 * j
	    η0 = .007/2
	    η0_bath = .007/2
	    δ = 1/5
	
	
	    p = WalkModel.auto_shift_bath(WalkModel.ExtendedModelParameters(v, u, 0, WalkModel.OhmicSpectralDensity(ω_c, J, α), N, WalkModel.LinearBathDiscretization(), true, true, 0., η0, η0_bath, η_coup, δ), 0)
	
	    p, WalkModel.ModelParameters(p), WalkModel.OhmicSpectralDensity(p)
	end
	
	prototype
	
end

# ╔═╡ 85ee3ff4-9937-4887-8ce1-057600942674
begin
	  using Plots
	  trans = WalkModel.Transmission(1, η_coup, full_prototype, n)
	  @show real(trans.peak_amplitudes[1])/real(trans.peak_amplitudes[end])
	  plot(0:.001:full_prototype.N+1, x-> trans(x), ylabel=L"I/I_0", xlabel=L"ω_L-ω_s", tickcolor="black", lw=2)
	  xticks!([full_prototype.δ - prototype.ω_A; collect(1:1:full_prototype.N) .- prototype.ε; trans.peak_positions], [L"δ-ε_A"; [L"%$n Ω_B-ε_%$n" for n in 1:1:full_prototype.N]])
	  vline!([full_prototype.δ - prototype.ω_A; collect(1:1:full_prototype.N) .- prototype.ε],legend=false, linestyle=:dash, color=:black)
	  #plot(x->1-full_prototype.spectral_density(x))
	  #hline!([(1-η_coup/(prototype.ω_A - prototype.ε[2])^2 * prototype.g[2]^2/prototype.η[3])])
	  #plot(x-> trans(x), .9, 1.1)
	
	  xlims!(0.15,.32)
	ylims!(0.6, 1.1)
end

# ╔═╡ 1433fea6-ea3f-4a72-b0a2-02acf5e36878
xlims!(0, full_prototype.N+.5)

# ╔═╡ 68782b87-ea76-4342-a33b-c42e4850ed5c
xlims!(n-0.15,n+.15)

# ╔═╡ 048ad6c8-7bef-4766-a67a-ae7e09b30c13
xlims!(full_prototype.N-0.05,full_prototype.N+.15)

# ╔═╡ Cell order:
# ╠═9d844d42-ca07-403b-864b-8c70be71b4f2
# ╠═2e29b1b6-3584-11ee-1981-0b7192ce1847
# ╠═40ba9793-d02f-4599-929a-4352f3712c82
# ╠═3fc99094-b156-4d6c-b176-9ea684745ece
# ╠═59923ada-370f-4a9c-9920-08a792bf70ba
# ╠═c61dec24-e44f-4b84-956a-aa5f4b11e119
# ╟─23cb9bd6-6bbe-4b41-834a-02d1de30d13d
# ╟─1433fea6-ea3f-4a72-b0a2-02acf5e36878
# ╠═2d11dd54-2314-4c47-ade9-7636384f1181
# ╟─85ee3ff4-9937-4887-8ce1-057600942674
# ╟─68782b87-ea76-4342-a33b-c42e4850ed5c
# ╠═048ad6c8-7bef-4766-a67a-ae7e09b30c13
