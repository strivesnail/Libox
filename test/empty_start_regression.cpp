#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT=INT64_MAX;
#else
#include "btreeolc.h"
#endif
using Key=int64_t;
int main(int argc,char**argv){
 const int threads=argc>1?std::stoi(argv[1]):1;
#ifdef LIBOX
 liboxns::LiBox<Key,Key> index(.5,.1,threads);
 auto put=[&](Key key){if(index.insertKeyValue(key,key^0x5a5a5a5a).status!=liboxns::InsertStatus::SUCCESS)throw std::runtime_error("insertion failed");};
 auto get=[&](Key key){auto r=index.searchKey(key);return r.status==liboxns::SearchStatus::SUCCESS&&r.value==(key^0x5a5a5a5a);};
 if(index.deleteKey(0).found)throw std::runtime_error("empty delete found a key");
#else
 ribtree::BTree<Key,Key> index(threads,.5,.1);
 auto put=[&](Key key){index.insert(key,key^0x5a5a5a5a);};
 auto get=[&](Key key){Key value;return index.lookup(key,value)&&value==(key^0x5a5a5a5a);};
#endif
 if(get(0))throw std::runtime_error("empty lookup succeeded");
 std::vector<Key> keys={42,0,1,-1,64,65,-64,-65,1000000,-1000000,Key(1)<<40,-(Key(1)<<40)};
 for(Key key:keys)put(key);
 for(Key key:keys)if(!get(key))throw std::runtime_error("growth lost keys");
 std::vector<Key> batch;for(Key i=0;i<8192;++i)batch.push_back((Key(1)<<42)+i*512);
 omp_set_dynamic(0);omp_set_num_threads(threads);
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<batch.size();++i)put(batch[i]);
 for(Key key:batch)if(!get(key))throw std::runtime_error("concurrent growth lost keys");
 for(Key i=0;i<1024;++i){put(std::numeric_limits<Key>::lowest()+i);put(std::numeric_limits<Key>::max()-1-i);}
 for(Key i=0;i<1024;++i)if(!get(std::numeric_limits<Key>::lowest()+i)||!get(std::numeric_limits<Key>::max()-1-i))throw std::runtime_error("extreme signed keys lost");
 bool rejected=false;try{put(std::numeric_limits<Key>::max());}catch(const std::out_of_range&){rejected=true;}
 if(!rejected)throw std::runtime_error("unrepresentable exclusive upper bound accepted");
 for(Key key:keys)if(!get(key))throw std::runtime_error("extreme growth corrupted old keys");
 // Race the very first insert without a directory, independently of the
 // earlier bidirectional/extreme-key checks.
#ifdef LIBOX
 liboxns::LiBox<Key,Key> fresh(.5,.1,threads);
#else
 ribtree::BTree<Key,Key> fresh(threads,.5,.1);
#endif
 #pragma omp parallel for schedule(static)
 for(Key i=0;i<4096;++i){
#ifdef LIBOX
  if(fresh.insertKeyValue(i*512,i^0x5a5a5a5a).status!=liboxns::InsertStatus::SUCCESS)throw std::runtime_error("concurrent first insert failed");
#else
  fresh.insert(i*512,i^0x5a5a5a5a);
#endif
 }
 for(Key i=0;i<4096;++i){
#ifdef LIBOX
  auto r=fresh.searchKey(i*512);if(r.status!=liboxns::SearchStatus::SUCCESS||r.value!=(i^0x5a5a5a5a))throw std::runtime_error("concurrent first insert mismatch");
#else
  Key v;if(!fresh.lookup(i*512,v)||v!=(i^0x5a5a5a5a))throw std::runtime_error("concurrent first insert mismatch");
#endif
 }
#ifdef LIBOX
 liboxns::LiBox<Key,Key> bulk(.5,.1,threads);
 std::vector<std::pair<Key,Key>> pairs;for(Key i=0;i<4096;++i)pairs.emplace_back(i*64,i);
 bulk.bulk_load(pairs.data(),pairs.size());
 for(auto [k,v]:pairs){auto r=bulk.searchKey(k);if(r.status!=liboxns::SearchStatus::SUCCESS||r.value!=v)throw std::runtime_error("unconfigured bulk mismatch");}
#else
 ribtree::BTree<Key,Key> bulk(threads,.5,.1);std::vector<Key> input;for(Key i=0;i<4096;++i)input.push_back(i*64);
 bulk.bulkLoad(input);
 for(Key k:input){Key v;if(!bulk.lookup(k,v)||v!=k)throw std::runtime_error("unconfigured key bulk mismatch");}
 ribtree::BTree<Key,Key> bulk_pairs(threads,.5,.1);std::vector<std::pair<Key,Key>> pairs;
 for(Key i=0;i<4096;++i)pairs.emplace_back(i*64,i);
 bulk_pairs.bulkLoad(pairs);
 for(auto [k,v]:pairs){Key got;if(!bulk_pairs.lookup(k,got)||got!=v)throw std::runtime_error("unconfigured pair bulk mismatch");}
#endif
 std::cerr<<"PASS empty startup, bidirectional/concurrent growth, extreme keys\n";
}
