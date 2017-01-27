# I/O without libuv, for use after STDOUT is finalized
raw_print(msg::AbstractString...) =
    ccall(:write, Cssize_t, (Cint, Cstring, Csize_t), 1, join(msg), length(join(msg)))
raw_println(msg::AbstractString...) = raw_print(msg..., "\n")

# safe version of `Base.print_with_color`, switching to raw I/O before finalizers are run
# (see `atexit` in `__init_logging__`)
function safe_print_with_color(color::Union{Int, Symbol}, io::IO, msg::AbstractString...)
    print_with_color(color, io, msg...)
end

const TRACE = haskey(ENV, "TRACE")
"Display a trace message. Only results in actual printing if the TRACE environment variable
is set."
@inline function trace(io::IO, msg...; prefix="TRACE: ", line=true)
    @static if TRACE
        safe_print_with_color(:cyan, io, prefix, chomp(string(msg...)), line?"\n":"")
    end
end
@inline trace(msg...; kwargs...) = trace(STDERR, msg...; kwargs...)

const DEBUG = TRACE || haskey(ENV, "DEBUG")
"Display a debug message. Only results in actual printing if the TRACE or DEBUG environment
variable is set."
@inline function debug(io::IO, msg...; prefix="DEBUG: ", line=true)
    @static if DEBUG
        safe_print_with_color(:green, io, prefix, chomp(string(msg...)), line?"\n":"")
    end
end
@inline debug(msg...; kwargs...) = debug(STDERR, msg...; kwargs...)

"Create an indented string from any value (instead of escaping endlines as \n)"
function repr_indented(ex; prefix=" "^7, abbrev=true)
    io = IOBuffer()
    print(io, ex)
    str = String(take!(io))

    # Limit output
    if abbrev && length(str) > 256
        if isa(ex, Array)
            T = eltype(ex)
            dims = join(size(ex), " by ")
            if method_exists(zero, (T,)) && zeros(ex) == ex
                str = "$T[$dims zeros]"
            else
                str = "$T[$dims elements]"
            end
        else
            if contains(strip(str), "\n")
                str = str[1:100] * "…\n\n[snip]\n\n…" * str[end-100:end]
            else
                str = str[1:100] * "…" * str[end-100:end]
            end
        end
    end

    lines = split(strip(str), '\n')
    if length(lines) > 1
        for i = 1:length(lines)
            lines[i] = prefix * lines[i]
        end

        lines[1] = "\"\n" * lines[1]
        lines[length(lines)] = lines[length(lines)] * "\""

        return join(lines, '\n')
    else
        return str
    end
end


function __init_logging__()
    if TRACE
        trace("CUDAdrv.jl is running in trace mode, this will generate a lot of additional output")
    elseif DEBUG
        debug("CUDAdrv.jl is running in debug mode, this will generate additional output")
        debug("Run with TRACE=1 to enable even more output")
    end
end
