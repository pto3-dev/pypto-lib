#include "orchestration_api.h"
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&) {return {.expected_arg_count=8};}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs& args) {
 const int grid=rt_available_cluster_count();
 uint32_t pipe_shape[1]={static_cast<uint32_t>(512*grid)};uint32_t offset[1]={0};
 Tensor pipe=args.tensor(6).ref().view(pipe_shape,offset);
 SIMPLER_SCOPE() {
  for (int64_t epoch=1;epoch<=3;++epoch) {
   CoreTaskArgs up;up.add_input(args.tensor(0).ref());up.add_input(args.tensor(1).ref());up.add_inout(args.tensor(3).ref());up.add_inout(pipe);up.launch_spec.set_block_num(grid);up.set_allow_early_resolve(true);
   MixedKernels mixed={0,1,1};rt_submit_task(mixed,up);
   CoreTaskArgs down;down.add_input(args.tensor(3).ref());down.add_input(args.tensor(2).ref());down.add_inout(args.tensor(4).ref());down.launch_spec.set_block_num(grid);down.set_allow_early_resolve(true);rt_submit_aic_task(2,down);
   CoreTaskArgs region;region.add_input(args.tensor(0).ref());region.add_input(args.tensor(1).ref());region.add_inout(args.tensor(3).ref());region.add_inout(pipe);region.add_input(args.tensor(2).ref());region.add_inout(args.tensor(5).ref());region.add_inout(args.tensor(7).ref());region.add_scalar(epoch);region.launch_spec.set_block_num(grid);region.launch_spec.set_static_block_mapping(true);
   MixedKernels compiled={3,4,4};rt_submit_task(compiled,region);
  }
 }
}
