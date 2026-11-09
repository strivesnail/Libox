#include <cstdint>
#include <iostream>
#include <stdexcept>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT = INT64_MAX;
using namespace liboxns;
#else
#include "rib_segment.h"
using namespace ribtree;
#endif
int main() {
    ThreadIdManager::initialize(1);
    Segment<int64_t,int64_t> exact(100,200,10,1), partial(100,201,10,1);
    if(exact.getBoxCount()!=10 || exact.getBoxUpper(9)!=200 || partial.getBoxCount()!=11)
        throw std::runtime_error("half-open bucket count invariant failed");
    if(exact.insertKeyValue(199,77).status!=InsertStatus::SUCCESS ||
       exact.insertKeyValue(200,88).status!=InsertStatus::OUT_OF_RANGE)
        throw std::runtime_error("half-open insertion boundary failed");
    unsigned rejected=0;
    try { Segment<int64_t,int64_t> empty(100,100,10,1); } catch(const std::invalid_argument&) { ++rejected; }
    try { Segment<int64_t,int64_t> zero(100,200,0,1); } catch(const std::invalid_argument&) { ++rejected; }
    if(rejected!=2)throw std::runtime_error("invalid segment parameters accepted");
    std::cout<<"Half-open segment boundary regression passed\n";
}
