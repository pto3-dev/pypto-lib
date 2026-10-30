#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include "callable.h"
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ifstream in(argv[1],std::ios::binary);std::vector<char>b((std::istreambuf_iterator<char>(in)),{});
 if(b.size()<offsetof(ChipCallable,storage_))return 3;
 const auto*c=reinterpret_cast<const ChipCallable*>(b.data());
 if(c->child_count()<0||c->child_count()>CHIP_MAX_CHILDREN)return 4;
 for(int i=0;i<c->child_count();++i){
  const auto&child=c->child(i);auto start=reinterpret_cast<const char*>(child.binary_data());
  if(start<b.data()||start+child.binary_size()>b.data()+b.size())return 5;
  std::string file=std::string(argv[2])+"/"+std::to_string(c->child_func_id(i))+".bin";
  std::ofstream out(file,std::ios::binary);out.write(start,child.binary_size());
  std::cout<<c->child_func_id(i)<<" "<<child.binary_size()<<"\n";
 }
}
