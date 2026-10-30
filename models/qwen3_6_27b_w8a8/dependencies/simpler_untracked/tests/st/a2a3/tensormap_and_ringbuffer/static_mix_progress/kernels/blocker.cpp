#include "aicore/aicore.h"
#include "async_kernel_api.h"
#include "tensor.h"
static __aicore__ inline bool wait_for(__gm__ int64_t* args, volatile __gm__ int64_t* p,int64_t value) {
    const uint64_t start=get_sys_cnt_aicore();
    do {
        if (get_sys_cnt_aicore()-start>1000000) {
            auto ctx=get_async_ctx(args);pto2::detail::defer_error(ctx,SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);pto2::detail::defer_flush(ctx);return false;
        }
        dcci((__gm__ void*)p,cache_line_t::SINGLE_CACHE_LINE);
    } while (*p!=value);
    return true;
}
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    auto* state_desc=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* trace_desc=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* state=reinterpret_cast<volatile __gm__ int64_t*>(state_desc->buffer.addr);
    auto* trace=reinterpret_cast<__gm__ int64_t*>(trace_desc->buffer.addr);
    const int block=get_block_idx(args),grid=get_block_num(args);

    trace[0]=get_sys_cnt_aicore();
    *state=1;dsb(DSB_DDR);dcci((__gm__ void*)state,SINGLE_CACHE_LINE,CACHELINE_OUT);
    if (!wait_for(args,state+16,1)) return;
    trace[1]=get_sys_cnt_aicore();trace[2]=get_physical_core_id();
    dsb(DSB_DDR);dcci((__gm__ void*)trace,SINGLE_CACHE_LINE,CACHELINE_OUT);
}
