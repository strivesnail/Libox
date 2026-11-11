#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <iostream>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
#else
#include "rib_segment.h"
using namespace ribtree;
#endif
int main(){
 ThreadIdManager::initialize(1);
 std::mt19937_64 rng(42);
 const uint64_t width=(UINT64_C(1)<<61)+3;
 Segment<uint64_t,uint64_t> wide(0,UINT64_MAX,width,1);
 for(int i=0;i<1000000;++i){uint64_t k=rng();if(k==UINT64_MAX)--k;if(wide.getLogicalBoxIndex(k)!=k/width)throw std::runtime_error("wide integer bucket index");}
 Segment<int64_t,int64_t> s(-100,101,7,1);
 for(int64_t k=-100;k<101;++k)if(s.getLogicalBoxIndex(k)!=size_t(k+100)/7)throw std::runtime_error("segment index");
 if(s.insertKeyValue(-99,1234).status!=InsertStatus::SUCCESS)return 2;
 Segment<int64_t,int64_t> moved(std::move(s));auto found=moved.searchKey(-99);if(found.status!=SearchStatus::SUCCESS||found.value!=1234)return 3;
 Segment<int64_t,int64_t> assigned(0,20,3,1);assigned=std::move(moved);found=assigned.searchKey(-99);if(found.status!=SearchStatus::SUCCESS||found.value!=1234||assigned.getLogicalBoxIndex(100)!=200/7)return 4;
 Box<int64_t,int64_t> box;
 for(int64_t i=0;i<64;++i)if(box.insertKeyValue(i*256+9,i+100).status!=InsertStatus::SUCCESS)return 5;
 for(int64_t i=0;i<64;++i){auto r=box.searchKey(i*256+9);if(r.status!=SearchStatus::SUCCESS||r.value!=i+100)return 6;}
 if(box.searchKey(100*256+9).status==SearchStatus::SUCCESS)return 7;
 auto fp=[](uint64_t b){b^=b>>33;b*=UINT64_C(0xff51afd7ed558ccd);b^=b>>33;uint8_t f=b>>56;return f?f:uint8_t(1);};
 int64_t collision=100000;while(fp(collision)!=fp(9))++collision;
 if(box.searchKey(collision).status==SearchStatus::SUCCESS)throw std::runtime_error("fingerprint collision false hit");
 auto entries=box.getEntries();if(entries.size()!=64)return 8;
 for(int64_t i=0;i<64;i+=2)if(!box.deleteKey(i*256+9).found)return 9;
 Box<int64_t,int64_t> copied(box),copy_assigned;copy_assigned=box;Box<int64_t,int64_t> moved_box(std::move(copied));copy_assigned=std::move(moved_box);
 for(int64_t i=0;i<64;++i){auto r=copy_assigned.searchKey(i*256+9);if((i%2==0)==(r.status==SearchStatus::SUCCESS))return 10;if(i%2&&r.value!=i+100)return 11;}
 auto idx=copy_assigned.searchUpdateKey(265);if(idx==maxKey)return 12;copy_assigned.updateValue(idx,777);if(copy_assigned.searchKey(265).value!=777)return 13;
 std::cout<<"PASS wide integer indexing, segment bounds/moves, Box CRUD/copy/move, collision misses; Box="<<sizeof(box)<<" Segment="<<sizeof(assigned)<<"\n";
}
