function targeted_saddle(p, q;
    scale=1.0,
    rng=Random.default_rng())

    @assert 0 < p < 1
    @assert 0 < q < 1
    @assert scale > 0

    M = [
        p*q -p*(1-q);
        -(1-p)*q (1-p)*(1-q)
    ]

    # Random overall scale.
    return scale * exp(randn(rng)) * M
end

function targeted_222(;
    targets=nothing,
    noise=0.25,
    rng=Random.default_rng()
)

    if targets === nothing
        targets = [(rand(rng), rand(rng)) for _ in 1:3]
    end

    Δs = ntuple(3) do i
        p, q = targets[i]

        M = targeted_saddle(p, q; rng=rng)

        # Perturb while keeping the desired geometry approximately.
        M + noise .* randn(rng, 2, 2)
    end

    return differences_to_game(Δs; rng=rng)
end
function differences_to_game(Δs;
    rng=Random.default_rng(),
    baseline_scale=1.0)

    U = [zeros(2, 2, 2) for _ in 1:3]

    for i in 1:3
        Δ = Δs[i]
        opponents = filter(!=(i), 1:3)

        for a in 1:2, b in 1:2

            profile = [1, 1, 1]
            profile[opponents[1]] = a
            profile[opponents[2]] = b

            baseline = baseline_scale * randn(rng)

            # action 2
            profile[i] = 2
            U[i][profile...] = baseline

            # action 1 = action 2 + Δ
            profile[i] = 1
            U[i][profile...] = baseline + Δ[a, b]
        end
    end

    return (U[1], U[2], U[3])
end

function symmetric_222(;
    σ=1.0,
    symmetry=0.8,
    rng=Random.default_rng()
)

    # Base positive magnitudes.
    x = exp.(randn(rng, 4))

    # Checkerboard signs.
    S = [1.0 -1.0;
        -1.0 1.0]

    # Base difference tensor.
    Δ0 = S .* reshape(x, 2, 2)

    Δ = Vector{Matrix{Float64}}(undef, 3)

    for i in 1:3
        # Symmetric component + independent random component.
        Y = randn(rng, 2, 2)
        Y ./= maximum(abs, Y)

        Δ[i] =
            symmetry .* Δ0 .+
            (1 - symmetry) .* (S .* abs.(Y) .* exp.(randn(rng)))
    end

    U = ntuple(_ -> zeros(2, 2, 2), 3)

    for i in 1:3
        Ui = U[i]
        opp = filter(!=(i), 1:3)

        for a in 1:2, b in 1:2

            p = [1, 1, 1]
            p[opp[1]] = a
            p[opp[2]] = b

            # Random payoff baseline.
            baseline = σ * randn(rng)

            p[i] = 2
            Ui[p...] = baseline

            p[i] = 1
            Ui[p...] = baseline + σ * Δ[i][a, b]
        end
    end

    return U
end


function random_222(; bias=0.8, σ=1.0)

    U = ntuple(_ -> randn(2, 2, 2) .* σ, 3)

    # We bias payoff differences, rather than payoffs themselves.
    #
    # For each player, Δ_i(a_j,a_k) determines whether action 1
    # or action 2 is preferred at each pure profile of opponents.
    #
    # Checkerboard sign patterns tend to put the indifference
    # surface through the interior.

    for i in 1:3

        Ui = U[i]

        opp = filter(!=(i), 1:3)

        Δ = randn(2, 2)

        signs = rand(Bool) ?
                [1.0 -1.0;
            -1.0 1.0] :
                [-1.0 1.0;
            1.0 -1.0]

        # Interpolate between completely random signs and
        # checkerboard signs.
        random_signs = sign.(randn(2, 2))

        S = sign.((1-bias) .* random_signs .+
                  bias .* signs)

        # Make sure there are no zeros.
        S[S .== 0] .= 1

        Δ .= S .* abs.(Δ)

        # Replace Ui with payoffs consistent with these differences.
        for a in 1:2, b in 1:2

            p = [1, 1, 1]
            p[opp[1]] = a
            p[opp[2]] = b

            baseline = randn() * σ

            p[i] = 2
            Ui[p...] = baseline

            p[i] = 1
            Ui[p...] = baseline + Δ[a, b] * σ
        end
    end

    return U
end


open("/tmp/path.dat", "w") do io;
    g = random_222();
    println(g)
    redirect_stdout(io) do;
        explore_manifold( g );
    end
end
