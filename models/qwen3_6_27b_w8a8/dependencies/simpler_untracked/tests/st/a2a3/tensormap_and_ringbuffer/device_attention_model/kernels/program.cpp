#include "orchestration_api.h"
#include <stdexcept>
static Tensor shape2(const Tensor &t, uint32_t a, uint32_t b) {
    uint32_t dims[2] = {a, b};
    return t.reshape(dims, 2);
}
static Tensor control(const Tensor &storage, uint32_t offset, uint32_t bytes, DataType dtype) {
    Tensor v = storage;
    v.buffer.addr += offset;
    v.buffer.size = bytes;
    v.start_offset = 0;
    v.dtype = dtype;
    v.shapes[0] = bytes / get_element_size(dtype);
    v.refresh_row_major_derived();
    return v;
}
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return {.expected_arg_count = 17};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    if (rt_available_cluster_count() < 3) throw std::runtime_error("Need three resident cluster positions");
    const Tensor &q = args.tensor(0).ref();
    const Tensor &k = args.tensor(1).ref();
    const Tensor &v = args.tensor(2).ref();
    const Tensor &table = args.tensor(3).ref();
    const Tensor &lengths = args.tensor(4).ref();
    const Tensor &gate = args.tensor(5).ref();
    const Tensor &sp = args.tensor(14).ref();
    const Tensor &vp = args.tensor(15).ref();
    Tensor qf = shape2(q, 12, 256), kf = shape2(k, 12288, 512), vf = shape2(v, 12288, 512);
    Tensor rof = shape2(args.tensor(8).ref(), 12, 256), rflat = shape2(args.tensor(8).ref(), 1, 3072);
    Tensor of = shape2(args.tensor(12).ref(), 12, 256), flat = shape2(args.tensor(12).ref(), 1, 3072);
    Tensor image = control(args.tensor(16).ref(), 0, 8192, DataType::UINT8);
    Tensor frames = control(args.tensor(16).ref(), 8192, 4608, DataType::UINT8);
    Tensor events = control(args.tensor(16).ref(), 12800, 2560, DataType::INT64);
    Tensor trace = control(args.tensor(16).ref(), 15360, 2560, DataType::INT64);
    SIMPLER_SCOPE() {
        CoreTaskArgs scores;
        scores.add_input(q);
        scores.add_input(k);
        scores.add_input(table);
        scores.add_input(lengths);
        scores.add_inout(args.tensor(6).ref());
        scores.add_input(qf);
        scores.add_input(kf);
        scores.add_inout(sp);
        scores.launch_spec.set_block_num(2);
        MixedKernels sk = {23, 24, 24};
        rt_submit_task(sk, scores);
        CoreTaskArgs soft;
        soft.add_input(args.tensor(6).ref());
        soft.add_inout(args.tensor(7).ref());
        soft.launch_spec.set_block_num(2);
        rt_submit_aiv_task(25, soft);
        CoreTaskArgs values;
        values.add_input(args.tensor(7).ref());
        values.add_input(v);
        values.add_input(table);
        values.add_input(lengths);
        values.add_input(args.tensor(8).ref());
        values.add_input(vf);
        values.add_inout(rof);
        values.add_inout(vp);
        values.launch_spec.set_block_num(2);
        MixedKernels vk = {26, 27, 27};
        rt_submit_task(vk, values);
        CoreTaskArgs gated;
        gated.add_input(gate);
        gated.add_input(rflat);
        gated.add_inout(args.tensor(9).ref());
        gated.launch_spec.set_block_num(6);
        rt_submit_aiv_task(28, gated);
        for (int64_t epoch = 1; epoch <= 3; ++epoch) {
            CoreTaskArgs a;
            a.add_input(image);
            a.add_inout(frames);
            a.add_inout(events);
            a.add_inout(trace);
            a.add_input(q);
            a.add_input(k);
            a.add_input(table);
            a.add_input(lengths);
            a.add_inout(args.tensor(10).ref());
            a.add_input(qf);
            a.add_input(kf);
            a.add_inout(sp);
            a.add_inout(args.tensor(11).ref());
            a.add_input(v);
            a.add_input(args.tensor(12).ref());
            a.add_input(vf);
            a.add_inout(of);
            a.add_inout(vp);
            a.add_input(gate);
            a.add_input(flat);
            a.add_inout(args.tensor(13).ref());
            a.add_scalar(epoch);
            a.launch_spec.set_block_num(3);
            a.launch_spec.set_static_block_mapping(true);
            a.set_allow_early_resolve(true);
            MixedKernels program = {35, 36, 36};
            rt_submit_task(program, a);
        }
    }
}
