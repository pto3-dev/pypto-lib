#include "orchestration_api.h"
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&) {
    return {.expected_arg_count=7};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs& args) {
    const auto& x=args.tensor(0).ref(); const auto& w=args.tensor(1).ref();
    const auto& a=args.tensor(2).ref(); const auto& b=args.tensor(3).ref();
    const auto& state=args.tensor(4).ref();
    const int mode=args.scalar<int>(0); const int epoch=args.scalar<int>(1);
    SIMPLER_SCOPE() {
        if (mode==0) {
            for (int i=0;i<2;++i) {
                CoreTaskArgs call;call.add_input(x);call.add_input(w);call.add_inout(i==0?a:b);
                call.launch_spec.set_block_num(rt_available_cluster_count());
                call.launch_spec.set_static_block_mapping(true);
                rt_submit_aic_task(0,call);
            }
        } else {
            CoreTaskArgs call;call.add_input(x);call.add_input(w);call.add_inout(a);call.add_inout(b);
            if (mode==2) {call.add_inout(state);call.add_scalar(static_cast<int64_t>(epoch));}
            call.launch_spec.set_block_num(rt_available_cluster_count());
            call.launch_spec.set_static_block_mapping(true);
            rt_submit_aic_task(mode,call);
        }
    }
}
