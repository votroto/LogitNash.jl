using Revise
using LogitNash
using Random
using LinearAlgebra


# new validate 44

v44 = [
([0.8591377879155269 -0.07371999080124683 0.5680732946309738 -1.2644799257064592; 1.9287461275801556 1.7420588726003674 -0.18617557836419837 -0.23988669854262043; 0.9695785746626668 0.5199948952559921 0.6447756292665288 0.19839865565386075; -1.1362481450931134 0.21572859130031394 0.7794847381841444 -0.6584615954266079], [0.8591377879155269 1.9287461275801556 0.9695785746626668 -1.1362481450931134; -0.07371999080124683 1.7420588726003674 0.5199948952559921 0.21572859130031394; 0.5680732946309738 -0.18617557836419837 0.6447756292665288 0.7794847381841444; -1.2644799257064592 -0.23988669854262043 0.19839865565386075 -0.6584615954266079])
([-0.4692754514816902 -0.13404308547900645 -0.05377243053031965 -0.5567407729841541; 1.555213181197249 -0.31958632004012427 -0.30435092233342353 0.571583301378112; 0.8995943311189669 0.5753181416288097 -1.1552145818058486 -0.2975159991075447; 1.0910453377501923 0.1197818353569531 -1.8829162458832538 -1.8242016907775354], [-0.4692754514816902 1.555213181197249 0.8995943311189669 1.0910453377501923; -0.13404308547900645 -0.31958632004012427 0.5753181416288097 0.1197818353569531; -0.05377243053031965 -0.30435092233342353 -1.1552145818058486 -1.8829162458832538; -0.5567407729841541 0.571583301378112 -0.2975159991075447 -1.8242016907775354])
([-0.54 -0.08 -1.82 0.83; -0.39 -0.06 -1.32 0.61; -0.05 -0.01 -0.17 0.08; 0.21 0.03 0.7 -0.32], [-0.12 -0.55 -1.06 -1.44; -0.29 0.0 -0.69 -0.51; 0.84 0.21 -0.74 0.41; -0.05 -0.62 -1.19 1.87])
([1.1 -1.9 0.1 -0.22; -0.05 -0.37 0.96 -1.48; 0.96 -1.52 -1.01 -0.97; 0.15 -0.36 -1.08 -1.61], [1.1 -0.05 0.96 0.15; -1.9 -0.37 -1.52 -0.36; 0.1 0.96 -1.01 -1.08; -0.22 -1.48 -0.97 -1.61])
([-2.35 0.37 -0.05 2.44; 1.45 0.63 -0.53 -0.86; 0.74 0.51 -0.55 -0.48; -0.01 -0.15 -0.02 0.12], [-2.35 1.45 0.74 -0.01; 0.37 0.63 0.51 -0.15; -0.05 -0.53 -0.55 -0.02; 2.44 -0.86 -0.48 0.12])
([-2.25 -0.29 -0.46 0.28; 0.11 -1.65 -0.06 2.18; -0.19 -0.6 -0.24 0.82; 0.42 0.78 -1.3 -0.91], [-2.25 0.11 -0.19 0.42; -0.29 -1.65 -0.6 0.78; -0.46 -0.06 -0.24 -1.3; 0.28 2.18 0.82 -0.91])
([-0.56 0.31 0.24 0.95; -0.39 -0.67 -0.28 -0.48; -0.03 0.6 -0.8 -0.28; -1.12 0.45 -0.41 0.47], [-0.56 -0.39 -0.03 -1.12; 0.31 -0.67 0.6 0.45; 0.24 -0.28 -0.8 -0.41; 0.95 -0.48 -0.28 0.47])
([0.27 1.31 -0.15 0.69; -0.13 -1.48 -1.85 -0.66; 0.22 1.22 -0.14 -0.52; -0.37 -0.64 -0.06 1.01], [0.27 -0.13 0.22 -0.37; 1.31 -1.48 1.22 -0.64; -0.15 -1.85 -0.14 -0.06; 0.69 -0.66 -0.52 1.01])
([1.0798 0.1135 -0.4242 -0.9456; -0.4692 1.4848 1.6058 -1.9951; -0.8981 1.5885 -0.081 0.2403; -2.3037 0.404 -0.0271 0.89], [1.0798 -0.4692 -0.8981 -2.3037; 0.1135 1.4848 1.5885 0.404; -0.4242 1.6058 -0.081 -0.0271; -0.9456 -1.9951 0.2403 0.89])
([-0.0016 -0.5519 -1.8766 -0.4486; -0.5819 0.8821 -1.2672 0.6328; -0.4555 0.1938 -0.8078 0.0936; 1.3418 0.8492 -0.5132 -0.412], [-0.0016 -0.5819 -0.4555 1.3418; -0.5519 0.8821 0.1938 0.8492; -1.8766 -1.2672 -0.8078 -0.5132; -0.4486 0.6328 0.0936 -0.412])
([0.904 1.1727 0.0832 1.0949; -1.6243 -0.5436 -0.6434 0.756; -1.3296 0.4858 -2.419 -0.846; 0.7548 0.0446 -1.6459 -1.2662], [0.904 -1.6243 -1.3296 0.7548; 1.1727 -0.5436 0.4858 0.0446; 0.0832 -0.6434 -2.419 -1.6459; 1.0949 0.756 -0.846 -1.2662])
([-2.1818 1.2916 0.1024 2.3842; 0.0343 -0.0047 0.6165 0.4905; 0.4616 0.7695 -1.4964 1.7826; 0.538 0.1685 0.1925 0.0633], [-2.1818 0.0343 0.4616 0.538; 1.2916 -0.0047 0.7695 0.1685; 0.1024 0.6165 -1.4964 0.1925; 2.3842 0.4905 1.7826 0.0633])
([0.2 2.65 -0.44 -2.28; -0.06 -0.72 0.12 0.62; 0.37 4.77 -0.79 -4.11; 0.3 3.95 -0.65 -3.4], [-0.63 -0.26 -1.13 0.48; 0.3 1.07 0.29 -0.51; 1.51 0.14 0.44 -0.13; -0.62 0.5 -0.4 0.57])
([0.17 -0.6 0.19 0.95; -0.1 0.35 -0.11 -0.56; 0.02 -0.06 0.02 0.09; -0.66 2.29 -0.72 -3.64], [1.71 0.49 0.47 -1.17; -0.53 -0.36 0.79 -0.08; -0.46 0.76 -1.13 0.87; 0.48 -0.09 0.43 -0.52])
([-0.400244816160577 -0.08644486309010577 -0.0998052490292963 -1.007329250983897; 0.7381470423662099 -0.9987119231062709 -1.5006308227949485 0.4551890457357607; -0.7132308987892143 -0.5514048625680135 0.3356352301673285 -0.6159629245252535; 0.2593920145436618 -1.0755175276775628 -2.645847274602515 1.3151698957292135], [-0.400244816160577 0.7381470423662099 -0.7132308987892143 0.2593920145436618; -0.08644486309010577 -0.9987119231062709 -0.5514048625680135 -1.0755175276775628; -0.0998052490292963 -1.5006308227949485 0.3356352301673285 -2.645847274602515; -1.007329250983897 0.4551890457357607 -0.6159629245252535 1.3151698957292135])
([-1.2808567647130722 1.1307129692636464 0.6747081761224399 -0.3280131299679092; 1.1826409198737846 -0.9199538365486055 0.6918629108949372 0.7792528390224784; 0.5560421140629526 -0.021934984793911327 1.16572126694432 2.134115868548537; 0.8522098454979077 -2.5544305702255814 -0.6615908851555317 0.8393832512025897], [-1.2808567647130722 1.1826409198737846 0.5560421140629526 0.8522098454979077; 1.1307129692636464 -0.9199538365486055 -0.021934984793911327 -2.5544305702255814; 0.6747081761224399 0.6918629108949372 1.16572126694432 -0.6615908851555317; -0.3280131299679092 0.7792528390224784 2.134115868548537 0.8393832512025897])
([0.6591219058999981 1.828262342385325 -0.49873415958484546 -1.0503165312490743; -0.1215578862357515 1.204188370456944 0.3023176631487301 0.7796137470758207; -0.7677754393258263 0.5517483082012681 1.681194609616295 1.238851891221849; -0.11813966836287507 -1.3547014724349973 2.396046722308994 1.169696806278158], [0.6591219058999981 -0.1215578862357515 -0.7677754393258263 -0.11813966836287507; 1.828262342385325 1.204188370456944 0.5517483082012681 -1.3547014724349973; -0.49873415958484546 0.3023176631487301 1.681194609616295 2.396046722308994; -1.0503165312490743 0.7796137470758207 1.238851891221849 1.169696806278158])
([-0.38300503521566986 0.4578399158315638 1.6254144792482772 1.7596578383534596; 0.9082062931218967 0.8557251509052257 -0.012246960698252813 -0.08922888740250413; 0.879923274214855 -0.880149352299967 0.8332128118605971 -0.07236479974170824; 2.054233494053193 -1.3766839800336614 -0.03297746273517096 0.44674534623457196], [-0.3817685655286198 0.9080342214856261 0.8806929923706359 2.0536857211677417; 0.45794087599133104 0.8544437024842702 -0.8796403062592622 -1.3767727420786595; 1.6270832955290406 -0.012142875763171574 0.8333092321201601 -0.03281354140650593; 1.75991036910885 -0.09001616721154905 -0.07153779790940196 0.4461602213235704])
([0.7148734708419873 0.23129162192773117 0.6605656557422029 0.19317116694520486; 0.1785287830162914 0.05842157305888414 0.16569730978549663 0.04837459220086873; 0.7118762527469906 0.23035054285094464 0.6583672539150929 0.19188078400959105; 0.5736444675679355 0.18545753465185041 0.5299372162831619 0.15442180073123055], [0.7141366261578053 0.23102511409535312 0.6601249600754014 0.19242242848517455; 0.17837265597985455 0.05770403265961473 0.16488195408874823 0.048062091176667585; 0.711165966989528 0.2300640978861971 0.6573789781539767 0.19162199138333164; 0.5730864358051073 0.1853949991764151 0.5297426944630682 0.15441678758703528])
([-0.0927809845323594 0.24894368297893316 -0.062351407845763505 0.0504308431605516; 0.010929104203135873 -0.029324235625461253 0.007344662669761864 -0.005940483847328005; -0.706790450034292 1.8964124881053348 -0.4749828839788686 0.3841739609976161; -0.2731231883505448 0.7328257267114866 -0.1835463957923474 0.14845534076448816], [0.17616572676117728 0.10400645634726532 -0.23772691867190407 -0.6339418555778964; 2.3785758300231308 -1.7899999376310234 1.8186996101697361 1.5155137613973328; -0.26247996554551584 0.23409229899527714 1.8299156187654133 1.3400106384652557; -0.749251888300743 -1.2768553977311135 -0.6256539278524198 -1.2839015522212163])
([-0.16720039195109107 -2.1709246203289867 -0.9619753362497571 2.7218691999911817; -0.9264834204847395 -0.36222640457695987 -0.4245645919169493 1.31702472235412; -0.8622952007781433 0.9040368327837632 0.3464840405356701 -0.12764170306273714; -0.718810770577864 -0.5665929476652187 0.08465428267610343 -1.3809853285047942], [-0.16720039195109107 -0.9264834204847395 -0.8622952007781433 -0.718810770577864; -2.1709246203289867 -0.36222640457695987 0.9040368327837632 -0.5665929476652187; -0.9619753362497571 -0.4245645919169493 0.3464840405356701 0.08465428267610343; 2.7218691999911817 1.31702472235412 -0.12764170306273714 -1.3809853285047942])
([1.5439789137316915 0.9151218139479103 -0.44070182986644824 -0.05920047879504492; -2.0622253845585417 1.2642615020513903 -0.5031095823158216 -0.009390937954644718; -1.7987888811955246 0.6667343787909097 -0.006107238538732652 0.11432501215537996; 0.11677697849519812 -0.04093845593210622 0.40572727460707997 1.1468312553626372], [1.5439789137316915 -2.0622253845585417 -1.7987888811955246 0.11677697849519812; 0.9151218139479103 1.2642615020513903 0.6667343787909097 -0.04093845593210622; -0.44070182986644824 -0.5031095823158216 -0.006107238538732652 0.40572727460707997; -0.05920047879504492 -0.009390937954644718 0.11432501215537996 1.1468312553626372])
([-1.739825316057326 -0.5067868310620196 0.5005736492703067 -0.6717703561315397; 2.5110124070186943 -0.8660186813319625 -0.7334597429095839 0.08036276820660875; 0.9880588600806524 -1.2413770225259826 0.7685850753939335 -0.518869578537665; 1.8882623815143011 -0.2608400666366337 -1.0195435220948217 0.5250953649794429], [-1.739825316057326 2.5110124070186943 0.9880588600806524 1.8882623815143011; -0.5067868310620196 -0.8660186813319625 -1.2413770225259826 -0.2608400666366337; 0.5005736492703067 -0.7334597429095839 0.7685850753939335 -1.0195435220948217; -0.6717703561315397 0.08036276820660875 -0.518869578537665 0.5250953649794429])
([0.21449104719793238 -0.1189243695669005 0.7574888921336143 0.018199275377704188; 0.4370548332342638 -0.2423246619733632 1.5434871793168967 0.03708351173206273; 0.15294814889023078 -0.08480196456129972 0.5401461989685545 0.012977443658031818; -1.5493429872906204 0.8590318356570152 -5.471604145299121 -0.13145965786719627], [1.802562611827338 -1.4065410654232444 0.17007838867996125 -0.28325213741986277; 1.546389391252335 -0.11941331623522107 1.720150804174306 1.3319075213670268; -0.4084612101706964 1.2015513774407027 0.18647525917652807 -0.29380273119400957; -2.106132537052759 0.6247456539437809 -1.3287926700432169 -0.37350557056667694])
]


for (i,game) in enumerate(v44)
    letter = ('a':'z')[i]
    open(io -> write(io, export_gambit_format_buf(game)), "/tmp/tst44.$letter.nfg", "w")
end





function export_gambit_format_buf(payoffs)
    players = eachindex(payoffs)

    player_string = join(["\"p$i\"" for i in players], " ")
    dim_string = join(size(first(payoffs)), " ")

    ioin = IOBuffer()

    println(ioin, "NFG 1 R \"Exported Game\"")
    println(ioin, "{ $player_string } { $dim_string }")

    for i in eachindex(payoffs[1])
        for p in eachindex(payoffs)
            print(ioin, payoffs[p][i])
            print(ioin, " ")
        end
    end

    seekstart(ioin)
    ioin
end

function parse_gambit_output(line, payoffs::NTuple{N}) where N
    players = eachindex(payoffs)
    dims = size(first(payoffs))

    ne = parse.(Float64, split(strip(line), ",")[2:end])
    nes = Vector{Vector{Float64}}(undef, length(players))
    offset = 1
    for (i, p) in enumerate(players)
        nes[i] = @view ne[offset:(offset+dims[p]-1)]
        offset += dims[p]
    end

    return ntuple(i -> nes[i], N)
end

function gambit_equilibrium(
    in::IOBuffer
)
    open(pipeline(`gambit-logit -q -e -l1e6`; stdin=in), "r", stdout) do ioout
        zz = read(ioout, String)
        return zz
    end
end

function covariant_game(actions::NTuple{N,Int}, r) where {N}
    @assert -1/(N-1) <= r <= 1

    Σ = Matrix{Float64}(I, N, N)
    for i in 1:N, j in 1:N
        if i != j
            Σ[i, j] = r
        end
    end

    L = cholesky(Symmetric(Σ)).L

    G = Array{Float64}(undef, actions..., N)

    for I in CartesianIndices(actions)
        @views G[Tuple(I)..., :] .= L * randn(N)
    end

    slices = eachslice(G, dims=N)
    ntuple(i -> Array(slices[i]), N)
end

using Random
Random.get_tls_seed()

#Random.seed!(3462315634)
#=
Us = (
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2),
    randn(2, 2, 2, 2, 2)
)

nash(Us; stop_eps=0.0)
io = export_gambit_format_buf(Us)
gambit_equilibrium(io)
=#

function runtest(n, seed)
    Random.seed!(seed)
    Us = (
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n),
        randn(n, n, n, n, n)
    )

    tj = @timed pi, status = nash(Us; stop_eps=0.0)

    if status.regret >= 1e-4 || status.t <= 100 || status.stall == true # !all(norm.(pi .- pig) .<= 1e-5)
        @show status
        println(seed)
    end
    #
    return true
    io = export_gambit_format_buf(Us)
    tc = @timed pigstr = gambit_equilibrium(io)
    pig = parse_gambit_output(pigstr, Us)

    agree = all(norm.(pi .- pig) .<= 1e-5)
    eqg = check_equilibrium(Us, pig)
    eqj = check_equilibrium(Us, pi)

    println("$(tj.time) $(tc.time) $agree $eqg $eqj")
    pi, pig
end

function print_strats(ss)
    for p in eachindex(ss)
        println(round.(ss[p]; digits=5))
    end
end

function unilateral_deviations_simple(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    x::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    T = promote_type(P, X)

    result = ntuple(i -> zeros(T, size(payoffs[i], i)), N)
    for i in CartesianIndices(first(payoffs))
        @simd for p in 1:N
            @inbounds w = prod(x[z][i[z]] for z in 1:N if z != p)
            @inbounds result[p][i[p]] += w * payoffs[p][i]
        end
    end
    result
end

function check_equilibrium(
    payoffs::NTuple{N,<:AbstractArray{P,N}},
    pi::NTuple{N,<:AbstractVector{X}}
) where {N,P,X}
    deviations = unilateral_deviations_simple(payoffs, pi)
    max_deviation_incentive(deviations, pi)
end



#_pi,_status = nash(ntuple(_ -> randn(ntuple(_ -> 2, 2)...), 2))

#for D = 2:100
#    for _ = 1:10
#        Us = ntuple(_ -> rand(ntuple(_ -> D, 2)...), 2);
#        t = @timed pi,status = nash(Us)
#        println("$D $(t.time)")
#    end
#end



A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi,status = nash(Us)


Random.seed!(3462345634)


A = 5
D = 5
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

@time pi,status = nash(Us)
print_strats(pi)
@show status

#=
A = 100
D = 2
Us = ntuple(_ -> randn(ntuple(_ -> A, D)...), D);

pi,status = nash(Us)

Us = ntuple(_ -> rand(ntuple(_ -> A, D)...), D);

@time pi,status = nash(Us)
#print_strats(pi)



#nash((randn(3,4,2), randn(3,4,2),  randn(3,4,2)))
=#
#=


A::Matrix{Float64} = zeros(2,2) #*floatmin()
B::Matrix{Float64} = zeros(2,2) #*floatmin()

payoffs = (A, B)

@show pi, status = nash(payoffs; stop_eps=-1.0, stop_iters=10000, stop_t=1e6)


print_strats(pi)
=#
#=
io = export_gambit_format_buf(payoffs)
tc = @timed pigstr = gambit_equilibrium(io)
pig = parse_gambit_output(pigstr, payoffs)

println()
print_strats(pig)



=#


#=
// An implementation of the Blotto: https://en.wikipedia.org/wiki/Blotto_game
// This version supports n >= 2 players. Each player distributes M coins on N
// fields. Each field is won by at most one player: the one with the most
// coins on the specific field; if there is a draw, the field is considered
// drawn (not won by any player), and hence ignored in the scoring. The winner
// is the player with the most won fields: all player have won the same number
// of fields, they each receive 0. Otherwise, the winners share 1 / (number of
// winners) and losers share -1 / (number of losers), reducing to {-1,0,1} in
// the 2-player case.
//
// Parameters:
//   "coins"      int    number of coins each player starts with (default: 10)
//   "fields"     int    number of fields (default: 3)
//   "players"    int    number of players (default: 2)
=#

#=
using Combinatorics

function generate_blotto_actions(coins::Int, fields::Int)
    actions = Vector{Vector{Int}}()
    for c in combinations(1:(coins + fields - 1), fields - 1)
        pushfirst!(c, 0)
        push!(c, coins + fields)
        push!(actions, [c[i+1] - c[i] - 1 for i in 1:fields])
    end
    return actions
end
=#
function evaluate_blotto_game(allocations::Vector{NTuple{N,Int}}) where N
    num_players = length(allocations)
    num_fields = length(allocations[1])
    points = zeros(Int, num_players)

    for f in 1:num_fields
        field_allocs = [allocations[p][f] for p in 1:num_players]
        max_coins = maximum(field_allocs)
        winners = findall(x -> x == max_coins, field_allocs)

        if length(winners) == 1
            points[winners[1]] += 1
        end
    end

    max_points = maximum(points)
    game_winners = findall(x -> x == max_points, points)
    num_winners = length(game_winners)

    utilities = zeros(num_players)
    if num_winners == num_players
        return utilities
    end

    num_losers = num_players - num_winners
    for p in 1:num_players
        if p in game_winners
            utilities[p] = (num_losers)
        else
            utilities[p] = (-num_winners)
        end
    end

    return utilities
end

# Generates just the N tensors of type Int8 for `players`
function generate_blotto_tensors(players::Int, coins::Int, fields::Int)
    actions = generate_blotto_actionsa(coins, fields)
    num_actions = length(actions)

    dims = ntuple(_ -> num_actions, players)
    tensors = ntuple(_ -> zeros(dims) , players)

    for idx in CartesianIndices(dims)
        allocs = [actions[idx.I[p]] for p in 1:players]
        utils = evaluate_blotto_game(allocs)

        for p in 1:players
            tensors[p][idx] = utils[p]
        end
    end

    return tensors
end



# Actions: 1 = Option A, 2 = Option B
# Payoffs: 1 if your choice is unique, 0 if you are in the majority.

p1_minority = Float64[0 0; 1 0;;; 0 1; 0 0]
p2_minority = Float64[0 1; 0 0;;; 0 0; 1 0]
p3_minority = Float64[0 0; 0 1;;; 1 0; 0 0]

three_player_minority = (p1_minority, p2_minority, p3_minority)


# Actions: 1 = Volunteer, 2 = Ignore
# Payoffs:
#   If anyone volunteers: Volunteers get 1, Ignorers get 2.
#   If nobody volunteers: Everyone gets 0.

p1_volunteer = Float64[1 1; 2 2;;; 1 1; 2 0]
p2_volunteer = Float64[1 2; 1 2;;; 1 2; 1 0]
p3_volunteer = Float64[1 1; 1 1;;; 2 2; 2 0]

three_player_volunteer = (p1_volunteer, p2_volunteer, p3_volunteer)



blotto = (Float64[0 -1 -1; 2 -1 -2; 2 1 -1;;; -1 2 1; -1 0 -1; 1 2 -1;;; -1 1 2; -2 -1 2; -1 -1 0], Float64[0 2 2; -1 -1 1; -1 -2 -1;;; -1 -1 1; 2 0 2; 1 -1 -1;;; -1 -2 -1; 1 -1 -1; 2 2 0], Float64[0 -1 -1; -1 2 1; -1 1 2;;; 2 -1 -2; -1 0 -1; -2 -1 2;;; 2 1 -1; 1 2 -1; -1 -1 0])



#=
#tensors = generate_blotto_tensors(4, 10, 3)

tensors = generate_blotto_tensors(4, 10, 3);
tensors_lazy = generate_blotto_lazy_tensors(4, 10, 3);
tensor = tensors[1];
tensor_uncompressed = Float64.(tensor);
tensor_lazy = tensors_lazy[1];

@benchmark dot($tensor_lazy, prod) setup=(prod=randn(66,66,66,66))

=#
#println("Generated $(length(tensors)) tensors of shape $(size(tensors[1])) with type $(eltype(tensors[1]))")
#println("Approximate memory per tensor: $(Base.summarysize(tensors[1]) / 1024^2 |> x -> round(x, digits=2)) MB")
















# Returns the number of fields strictly won by player p
@inline function blotto_score(p::Int, actions::NTuple{N, NTuple{K, Int}}) where {N, K}
    score = 0
    for k in 1:K
        won = true
        coins_p = actions[p][k]
        for p2 in 1:N
            if p != p2 && actions[p2][k] >= coins_p
                won = false
                break
            end
        end
        score += won ? 1 : 0
    end
    return score
end

# Returns an NTuple of payoffs for all N players given a specific action profile
@inline function blotto_payoffs(actions::NTuple{N, NTuple{K, Int}}) where {N, K}
    scores = ntuple(p -> blotto_score(p, actions), Val(N))

    max_score = -1
    for p in 1:N
        if scores[p] > max_score
            max_score = scores[p]
        end
    end

    num_winners = 0
    for p in 1:N
        if scores[p] == max_score
            num_winners += 1
        end
    end

    if num_winners == N
        return ntuple(p -> 0.0, Val(N))
    else
        win_val = 1.0 / num_winners
        lose_val = -1.0 / (N - num_winners)
        return ntuple(p -> (scores[p] == max_score ? win_val : lose_val), Val(N))
    end
end




@generated function unilateral_derivatives_blotto!(
    results::NTuple{N, NTuple{N, Matrix{T}}},
    actions::Vector{NTuple{K, Int}},
    pi::NTuple{N, Vector{T}}
) where {N, T, K}

    # 1. Build the innermost compute payload dynamically
    updates = Expr(:block)

    # Create the profile tuple: a = (actions[i_1], actions[i_2], ..., actions[i_N])
    tuple_expr = Expr(:tuple, [:(actions[$(Symbol("i_", p))]) for p in 1:N]...)
    push!(updates.args, :( a = $tuple_expr ))

    # Evaluate Blotto algebra
    push!(updates.args, :( u = blotto_payoffs(a) ))

    # Update results for all p, q pairs concurrently
    for p in 1:N
        for q in 1:N
            if p != q
                # weight = prod(pi[r][i_r] for r != p,q)
                weight_expr = :(one(T))
                for r in 1:N
                    if r != p && r != q
                        i_r = Symbol("i_", r)
                        weight_expr = :( $weight_expr * pi[$r][$i_r] )
                    end
                end

                i_p = Symbol("i_", p)
                i_q = Symbol("i_", q)

                push!(updates.args, :(
                    @inbounds results[$p][$q][$i_p, $i_q] += T(u[$p]) * $weight_expr
                ))
            end
        end
    end

    # 2. Wrap the payload in D^N nested loops
    loop_expr = updates
    for n in 1:N
        i_n = Symbol("i_", n)
        loop_expr = quote
            for $i_n in 1:D
                $loop_expr
            end
        end
    end

    # 3. Return the fully generated syntax tree
    return quote
        D = length(actions)

        # Zero out the derivative matrices
        for p in 1:N
            for q in 1:N
                if p != q
                    fill!(results[p][q], zero(T))
                end
            end
        end

        # Execute the unrolled D^N loop
        @inbounds $loop_expr

        return nothing
    end
end









# 1. The generator for the action space (run once outside the loop)
function generate_blotto_actionsa(M::Int, K::Int)
    actions = NTuple{K, Int}[]
    function generate(fields_left, coins_left, current)
        if fields_left == 1
            push!(actions, tuple(current..., coins_left))
            return
        end
        for c in 0:coins_left
            generate(fields_left - 1, coins_left - c, (current..., c))
        end
    end
    generate(K, M, ())
    return actions
end

# 2. The Factory Function
function make_blotto_evaluator(M::Int, K::Int, ::Val{N}) where {N}
    # Pre-allocate the action map once.
    # For D=66, this is tiny and stays permanently in the L1 cache.
    const_actions = generate_blotto_actionsa(M, K)

    # Return a closure that perfectly matches your tracker's signature
    @inline function custom_derivatives!(
        results::NTuple{N, NTuple{N, Matrix{T}}},
        payoffs::Any, # Ignored! We bypass the 140MB memory wall.
        pi::NTuple{N, Vector{T}}
    ) where {T}
        # Forward to the @generated unrolled loop (from the previous Julia answer)
        unilateral_derivatives_blotto!(results, const_actions, pi)
        return nothing
    end

    return custom_derivatives!
end


# Setup your specific game parameters (e.g., 10 coins, 3 fields, 2 players)
blotto_eval! = make_blotto_evaluator(10, 3, Val(4))




A = 66
D = 4
dims = ntuple(_ -> A, D);

#blot = generate_blotto_tensors(4,10,3)


pi = ntuple(_ -> normalize(rand(A),1), D);

dudpi1 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);
dudpi2 = ntuple(p -> ntuple(q -> zeros(dims[p], dims[q]), D), D);

LogitNash.unilateral_derivatives!(dudpi1, blot, pi);
blotto_eval!(dudpi2, nothing, pi);

@show norm.(dudpi1[i] .- dudpi2[i] for i in eachindex(dudpi1))

nothing



#=


@benchmark LogitNash.unilateral_derivatives!($dudpi1, blot, $pi)
@benchmark blotto_eval!($dudpi2, nothing, $pi)

@benchmark unilateral_derivatives_old!($dudpi1, $Us, $pi)
@benchmark unilateral_derivatives!($dudpi2, $U_perm, $pi)


=#