# Stream management

export
    CuStream, CuDefaultStream, synchronize


typealias CuStream_t Ptr{Void}

type CuStream
    handle::CuStream_t
end

Base.unsafe_convert(::Type{CuStream_t}, s::CuStream) = s.handle

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

function CuStream(flags::Integer=0)
    handle_ref = Ref{CuStream_t}()
    @apicall(:cuStreamCreate, (Ptr{CuStream_t}, Cuint),
                              handle_ref, flags)
    obj = CuStream(handle_ref[])
    return obj
end

function Base.close(s::CuStream)
    trace("Closing CuStream at $(Base.pointer_from_objref(s))")
    @apicall(:cuStreamDestroy, (CuModule_t,), s)
end

function finalize(s::CuStream)
    close(s)
end

CuDefaultStream() = CuStream(convert(CuStream_t, C_NULL))

synchronize(s::CuStream) = @apicall(:cuStreamSynchronize, (CuStream_t,), s)
