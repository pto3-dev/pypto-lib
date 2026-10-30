#include "orchestration_api.h"
extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&) { return {.expected_arg_count=11}; }
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs& args) {
    const auto& x=args.tensor(0).ref();const auto& w=args.tensor(1).ref();const auto& c=args.tensor(2).ref();const auto& s=args.tensor(3).ref();
    const auto& p=args.tensor(4).ref();const auto& v=args.tensor(5).ref();const auto& k=args.tensor(6).ref();const auto& val=args.tensor(7).ref();
    const auto& slot=args.tensor(8).ref();const auto& temp=args.tensor(9).ref();const int mode=args.scalar<int>(0);
    uint32_t two[2]={2,256},flat[2]={1,512},cache[2]={24576,256};
    auto xf=x.reshape(two,2);auto tf=temp.reshape(two,2);auto flat_temp=temp.reshape(flat,2);auto kf=k.reshape(cache,2);
    SIMPLER_SCOPE() {
        if (mode==0) {
            CoreTaskArgs call;call.add_input(x);call.add_input(w);call.add_input(c);call.add_input(s);call.add_input(p);call.add_input(temp);call.add_input(xf);call.add_inout(tf);
            call.launch_spec.set_block_num(1);call.launch_spec.set_static_block_mapping(true);rt_submit_aiv_task(0,call);
            CoreTaskArgs append;append.add_input(flat_temp);append.add_input(v);append.add_inout(k);append.add_inout(val);append.add_input(slot);
            append.launch_spec.set_block_num(4);append.launch_spec.set_static_block_mapping(true);rt_submit_aiv_task(1,append);
        } else {
            CoreTaskArgs call;call.add_input(x);call.add_input(w);call.add_input(c);call.add_input(s);call.add_input(p);call.add_input(v);call.add_inout(k);call.add_inout(val);call.add_input(slot);call.add_input(xf);
            if(mode==1){call.add_inout(temp);call.add_inout(tf);call.add_input(flat_temp);}else{call.add_inout(kf);}
            call.launch_spec.set_block_num(1);call.launch_spec.set_static_block_mapping(true);rt_submit_aiv_task(mode==1?3:2,call);
        }
    }
}
