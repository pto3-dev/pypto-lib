// Production device plan ABI v2: constant scalar bindings.
#include "aicore/aicore.h"
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "tensor.h"
#include "intrinsic.h"
#include "device_program_types.h"
static __aicore__ __attribute__((always_inline)) void run_program(__gm__ int64_t *, uint32_t);
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    run_program(args, 1 + get_sub_block_id(args));
}
#include "device_program_body.h"
