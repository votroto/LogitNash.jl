using Mmap

# ASCII markers
const BYTE_SPACE     = UInt8(' ')  # 0x20
const BYTE_QUOTE     = UInt8('"')  # 0x22
const BYTE_LBRACE    = UInt8('{')  # 0x7b
const BYTE_RBRACE    = UInt8('}')  # 0x7d
const BYTE_BACKSLASH = UInt8('\\') # 0x5c
const BYTE_LF        = UInt8('\n') # 0x0a

function skip_whitespace(bytes::Vector{UInt8}, pos::Int, len::Int)
    @inbounds while pos <= len
        b = bytes[pos]
        if b <= BYTE_SPACE
            pos += 1
        else
            break
        end
    end
    return pos
end

function read_token(bytes::Vector{UInt8}, pos::Int, len::Int)
    pos = skip_whitespace(bytes, pos, len)
    start_pos = pos
    @inbounds while pos <= len
        b = bytes[pos]
        if b <= BYTE_SPACE || b == BYTE_QUOTE || b == BYTE_LBRACE || b == BYTE_RBRACE
            break
        end
        pos += 1
    end
    return String(bytes[start_pos:pos-1]), pos
end

function read_string(bytes::Vector{UInt8}, pos::Int, len::Int)
    pos = skip_whitespace(bytes, pos, len)
    bytes[pos] == BYTE_QUOTE || error("Expected '\"' at position $pos")
    pos += 1

    buf = IOBuffer()
    escaped = false
    @inbounds while pos <= len
        b = bytes[pos]
        if escaped
            write(buf, b)
            escaped = false
        elseif b == BYTE_BACKSLASH
            escaped = true
        elseif b == BYTE_QUOTE
            pos += 1
            return String(take!(buf)), pos
        else
            write(buf, b)
        end
        pos += 1
    end
    error("Unterminated double-quoted string")
end

function read_braced_block(bytes::Vector{UInt8}, pos::Int, len::Int)
    pos = skip_whitespace(bytes, pos, len)
    bytes[pos] == BYTE_LBRACE || error("Expected '{' at position $pos")
    pos += 1

    tokens = String[]
    @inbounds while pos <= len
        pos = skip_whitespace(bytes, pos, len)
        if pos > len break end

        b = bytes[pos]
        if b == BYTE_RBRACE
            pos += 1
            return tokens, pos
        elseif b == BYTE_QUOTE
            str, pos = read_string(bytes, pos, len)
            push!(tokens, str)
        else
            tok, pos = read_token(bytes, pos, len)
            push!(tokens, tok)
        end
    end
    error("Unterminated braced block")
end

function _parse_payoffs!(tensors::NTuple{N, Array{Float64, N}}, bytes::Vector{UInt8}, pos::Int, len::Int) where {N}
    num_profiles = length(tensors[1])

    @inbounds for p in 1:num_profiles
        for k in 1:N
            pos = skip_whitespace(bytes, pos, len)
            if pos > len
                error("Unexpected end of file while reading payoffs")
            end

            # 1. Manually find exactly where the float ends
            start_pos = pos
            while pos <= len
                b = bytes[pos]
                if b <= BYTE_SPACE
                    break
                end
                pos += 1
            end

            # 2. Call Julia's internal C-runtime float parser directly
            hasvalue, val = ccall(:jl_try_substrtod, Tuple{Bool, Float64},
                                  (Ptr{UInt8}, Csize_t, Csize_t),
                                  bytes, start_pos - 1, pos - start_pos)

            if !hasvalue
                ctx = String(bytes[max(1, start_pos - 15) : min(len, pos + 15)])
                error("Failed to parse float at position $start_pos. Context: '$(ctx)'")
            end

            tensors[k][p] = val
        end
    end
    return pos
end

function parse_nfg(bytes::Vector{UInt8})
    len = length(bytes)
    pos = 1

    tag, pos = read_token(bytes, pos, len)
    tag == "NFG" || error("Invalid file: does not start with NFG")

    _ver, pos = read_token(bytes, pos, len)
    _ver == "1" || error("Accepting only NFG version 1")

    _type, pos = read_token(bytes, pos, len)
    (_type == "D" || _type == "R") || error("Accepting only NFG D or R data type")

    _title, pos = read_string(bytes, pos, len)

    players, pos = read_braced_block(bytes, pos, len)
    N = length(players)

    sizes_str, pos = read_braced_block(bytes, pos, len)
    strategy_counts = Tuple(parse(Int, s) for s in sizes_str)

    pos = skip_whitespace(bytes, pos, len)
    if pos <= len && bytes[pos] == BYTE_QUOTE
        _comment, pos = read_string(bytes, pos, len)
    end

    tensors = ntuple(_ -> Array{Float64, N}(undef, strategy_counts...), N)
    _parse_payoffs!(tensors, bytes, pos, len)

    return tensors
end

function parse_nfg(io::Union{IOStream,AbstractString})
    bytes = Mmap.mmap(io)
    return parse_nfg(bytes)
end

function run_test()
    begin
        tensors = parse_nfg("/tmp/games/game1.game")
        ne, status = nash(tensors)
    end

    times_read = fill(NaN, 100)
    times_nash = fill(NaN, 100)

    for i in 1:100
        try
        @show i
        tr = @timed tensors = parse_nfg("/tmp/games/game$i.game")
        tn = @timed ne, status = nash(tensors)

        GC.gc()          # Force a garbage collection

        @assert status.stall == false
        @assert status.t >= 500000 || status.regret <= 1e-6

        times_read[i] = tr.time
        times_nash[i] = tn.time
        catch
        end
    end
    return times_read, times_nash
end