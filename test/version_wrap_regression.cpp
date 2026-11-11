#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
// White-box test only: set the version near wrap without billions of updates.
#define private public
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
using namespace liboxns;
#else
#include "rib_segment.h"
using namespace ribtree;
#endif
#undef private
int main(){using B=Box<int64_t,int64_t>;B src;src.insertKeyValue(42,99);auto verify=[](B&b){if(b.version_lock_.load()!=0||b.searchKey(42).status!=SearchStatus::SUCCESS||b.searchKey(42).value!=99)throw std::runtime_error("version wrap left lock set or lost payload");};B copy;copy.version_lock_.store(0x7fffffff);copy=src;verify(copy);B moved;moved.version_lock_.store(0x7fffffff);moved=std::move(copy);verify(moved);moved.version_lock_.store(0x7fffffff);moved.insertKeyValue(42,99);verify(moved);std::cout<<"PASS version wrap on copy/move/upsert\n";}
