#include "orchestration_api.h"
#include <stdexcept>
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return {.expected_arg_count = 12};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    constexpr int tile_count = 4;
    if (rt_available_cluster_count() < tile_count) {
        throw std::runtime_error("Device program requires four resident tile positions");
    }
    SIMPLER_SCOPE() {
        CoreTaskArgs project;
        project.add_input(args.tensor(4).ref());
        project.add_input(args.tensor(5).ref());
        project.add_inout(args.tensor(6).ref());
        project.launch_spec.set_block_num(tile_count);
        rt_submit_aic_task(0, project);
        CoreTaskArgs activation;
        activation.add_input(args.tensor(6).ref());
        activation.add_inout(args.tensor(7).ref());
        activation.launch_spec.set_block_num(tile_count * 2);
        rt_submit_aiv_task(1, activation);
        CoreTaskArgs consume;
        consume.add_input(args.tensor(7).ref());
        consume.add_inout(args.tensor(8).ref());
        consume.launch_spec.set_block_num(tile_count);
        rt_submit_aiv_task(2, consume);
        for (int64_t epoch = 1; epoch <= 3; ++epoch) {
            CoreTaskArgs program;
            program.add_input(args.tensor(0).ref());
            for (int i = 1; i < 4; ++i)
                program.add_inout(args.tensor(i).ref());
            program.add_input(args.tensor(4).ref());
            program.add_input(args.tensor(5).ref());
            for (int i = 9; i < 12; ++i)
                program.add_inout(args.tensor(i).ref());
            program.add_scalar(epoch);
            program.launch_spec.set_block_num(tile_count);
            program.launch_spec.set_static_block_mapping(true);
            program.set_allow_early_resolve(true);
            MixedKernels kernels = {3, 4, 4};
            rt_submit_task(kernels, program);
        }
    }
}
