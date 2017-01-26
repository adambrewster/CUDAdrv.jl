# Context management

export
    CuContext, destroy, CuCurrentContext, activate,
    synchronize, device

@enum(CUctx_flags, SCHED_AUTO           = 0x00,
                   SCHED_SPIN           = 0x01,
                   SCHED_YIELD          = 0x02,
                   SCHED_BLOCKING_SYNC  = 0x04,
                   MAP_HOST             = 0x08,
                   LMEM_RESIZE_TO_MAX   = 0x10)
Base.@deprecate_binding BLOCKING_SYNC SCHED_BLOCKING_SYNC

typealias CuContext_t Ptr{Void}


## construction and destruction

"""
Create a CUDA context for device. A context on the GPU is analogous to a process on the
CPU, with its own distinct address space and allocated resources. When a context is
destroyed, the system cleans up the resources allocated to it.

Contexts are unique instances which need to be `destroy`ed after use. For automatic
management, prefer the `do` block syntax, which implicitly calls `destroy`.
"""
type CuContext
    handle::CuContext_t
end

function Base.close(ctx::CuContext)
    trace("Closing CuContext at $(Base.pointer_from_objref(ctx))")
    @apicall(:cuCtxDestroy, (CuContext_t,), ctx)
end

function finalize(ctx::CuContext)
    close(ctx)
end

Base.unsafe_convert(::Type{CuContext_t}, ctx::CuContext) = ctx.handle

Base.:(==)(a::CuContext, b::CuContext) = a.handle == b.handle
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.handle, h)

"""
Mark a context for destruction.

This does not immediately destroy the context, as there might still be dependent resources
which have not been collected yet. The context will get freed as soon as all outstanding
instances have been finalized.
"""
function destroy(ctx::CuContext)
    close(ctx)
end

Base.deepcopy_internal(::CuContext, ::ObjectIdDict) =
    error("CuContext cannot be copied")

function CuContext(dev::CuDevice, flags::CUctx_flags=SCHED_AUTO)
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxCreate, (Ptr{CuContext_t}, Cuint, Cint),
                           handle_ref, flags, dev)
    CuContext(handle_ref[])
end

"Return the current context."
function CuCurrentContext()
    handle_ref = Ref{CuContext_t}()
    @apicall(:cuCtxGetCurrent, (Ptr{CuContext_t},), handle_ref)
    CuContext(handle_ref[])
end

activate(ctx::CuContext) = @apicall(:cuCtxSetCurrent, (CuContext_t,), ctx)

"Create a context, and activate it temporarily."
function CuContext(f::Function, args...)
    # NOTE: this could be implemented with context pushing and popping,
    #       but that functionality / our implementation of it hasn't been reliable
    old_ctx = CuCurrentContext()
    scope() do
      ctx = @! CuContext(args...)    # implicitly activates
      @defer activate(old_ctx)
      f(ctx)
    end
end


## context properties

function device(ctx::CuContext)
    if CuCurrentContext() != ctx
        # TODO: should we push and pop here?
        error("context should be active")
    end

    # TODO: cuCtxGetDevice returns the device ordinal, but as a CUDevice*?
    #       This can't be right...
    device_ref = Ref{Cint}()
    @apicall(:cuCtxGetDevice, (Ptr{Cint},), device_ref)
    return CuDevice(device_ref[])
end

synchronize(ctx::CuContext=CuCurrentContext()) =
    @apicall(:cuCtxSynchronize, (CuContext_t,), ctx)
