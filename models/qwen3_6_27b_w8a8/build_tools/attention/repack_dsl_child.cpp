#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include "callable.h"
static std::vector<uint8_t> read_file(const char *name) {
    std::ifstream in(name,std::ios::binary);if (!in) throw std::runtime_error("input missing");
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),{});
}
static const CoreCallable &find_child(const ChipCallable *p,int id) {
    for(int i=0;i<p->child_count();++i)if(p->child_func_id(i)==id)return p->child(i);
    throw std::runtime_error("child missing");
}
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    const int target=std::stoi(argv[4]);
    auto base_data=read_file(argv[1]), donor_data=read_file(argv[2]);
    auto *base=reinterpret_cast<const ChipCallable*>(base_data.data());
    auto *donor=reinterpret_cast<const ChipCallable*>(donor_data.data());
    std::vector<int32_t> ids;std::vector<std::vector<uint8_t>> children;
    for(int i=0;i<base->child_count();++i){
        int id=base->child_func_id(i);ids.push_back(id);
        const auto &child=id==target?find_child(donor,0):base->child(i);
        children.push_back(make_callable<CORE_MAX_TENSOR_ARGS>(base->child(i).signature_,base->child(i).sig_count(),child.binary_data(),child.binary_size()));
    }
    auto output=make_chip_callable(base->signature_,base->sig_count(),base->func_name(),base->binary_data(),base->binary_size(),
        ids.data(),children.data(),static_cast<int32_t>(children.size()),base->config_name());
    std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(output.data()),output.size());
}
