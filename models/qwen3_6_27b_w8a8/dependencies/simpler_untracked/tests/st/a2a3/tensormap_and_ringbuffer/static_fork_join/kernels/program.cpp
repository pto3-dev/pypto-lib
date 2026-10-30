#include "orchestration_api.h"
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&) {return {.expected_arg_count=6};}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs& args) {
    SIMPLER_SCOPE() {
        for (int epoch=1;epoch<=3;++epoch) {
            CoreTaskArgs call;call.add_input(args.tensor(0).ref());call.add_input(args.tensor(1).ref());
            for (int index=2;index<6;++index)call.add_inout(args.tensor(index).ref());
            call.add_scalar(static_cast<int64_t>(epoch));
            call.launch_spec.set_block_num(rt_available_cluster_count());
            MixedKernels kernels={0,1,2};rt_submit_task(kernels,call);
        }
    }
}
