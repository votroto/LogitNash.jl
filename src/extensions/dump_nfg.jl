
function dump_nfg(io, payoffs)
    players = eachindex(payoffs)

    player_string = join(["\"p$i\"" for i in players], " ")
    dim_string = join(size(first(payoffs)), " ")

    println(io, "NFG 1 R \"Exported Game\"")
    println(io, "{ $player_string } { $dim_string }")

    for i in eachindex(payoffs[1])
        for p in eachindex(payoffs)
            print(io, payoffs[p][i])
            print(io, " ")
        end
    end
end