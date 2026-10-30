#include "orchestration_api.h"
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&) {return {.expected_arg_count=2};}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs& args) {
    const int grid=rt_available_cluster_count();
    rt_scope_begin(ScopeMode::MANUAL);
    CoreTaskArgs blocker;blocker.add_inout(args.tensor(0).ref());blocker.add_inout(args.tensor(1).ref());
    blocker.launch_spec.set_static_block_mapping(true);rt_submit_aic_task(0,blocker);
    uint32_t shape[1]={1};
    TensorCreateInfo marker_info(shape,1,DataType::INT64);
    CoreTaskArgs marker;marker.add_input(args.tensor(0).ref());marker.add_output(marker_info);
    TaskOutputTensors signal=rt_submit_aiv_task(4,marker);
    uint32_t index[1]={0};
    if (get_tensor_data<int64_t>(signal.get_ref(0),1,index)!=1) __builtin_trap();
    CoreTaskArgs region;region.add_inout(args.tensor(0).ref());region.add_inout(args.tensor(1).ref());
    region.launch_spec.set_block_num(grid);region.launch_spec.set_static_block_mapping(true);
    MixedKernels kernels={1,2,3};rt_submit_task(kernels,region);
    rt_scope_end();
}
