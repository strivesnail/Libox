#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
#else
#include "rib_segment.h"
using namespace ribtree;
#endif
template<class T>void check(){Box<T,T>b;std::vector<T>keys;for(unsigned i=0;i<64;++i){T k=i==0?std::numeric_limits<T>::lowest():i==1?std::numeric_limits<T>::max():i%2?T(std::numeric_limits<T>::max()-i):T(std::numeric_limits<T>::lowest()+i);keys.push_back(k);if(b.insertKeyValue(k,T(i)).status!=InsertStatus::SUCCESS)throw std::runtime_error("typed insert");}for(unsigned i=0;i<64;++i){auto r=b.searchKey(keys[i]);if(r.status!=SearchStatus::SUCCESS||r.value!=T(i))throw std::runtime_error("typed value");}b.insertKeyValue(keys[0],T(0));if(b.getTotalCount()!=64||b.searchKey(keys[0]).value!=0)throw std::runtime_error("typed overwrite");b.deleteKey(keys[1]);Box<T,T>copy(b),moved(std::move(copy));if(moved.getTotalCount()!=63||moved.getEntries().size()!=63||moved.searchKey(keys[1]).status==SearchStatus::SUCCESS)throw std::runtime_error("typed delete/copy/move");}
int main(){check<int32_t>();check<uint32_t>();check<int64_t>();check<uint64_t>();std::cout<<"PASS signed/unsigned32/64 Box extremes and zero payload\n";}
