#include <bits/stdc++.h>
#include <omp.h>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT = INT64_MAX;
#else
#include "src/btreeolc.h"
#endif
using K=int64_t;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
 if(argc<3)return 1;
 std::string mode=argv[1],config=argv[2];
 const K n=argc>4?std::stoll(argv[4]):32768,step=4096,domain=n*step;
 if(n<256 || n%32)throw std::runtime_error("n must be a multiple of 32 and at least 256");
 std::mt19937_64 rng(argc>3?std::stoull(argv[3]):42);
 std::vector<K> initial,added;
 for(K i=0;i<n;i++)initial.push_back(i*step);
 std::unordered_set<K> seen(initial.begin(),initial.end());
 while(added.size()<size_t(n)){
 K k=mode=="uniform"?rng()%domain:domain/2+rng()%(domain/100);
 if(seen.insert(k).second)added.push_back(k);
 }
 if(mode=="hot_sorted")std::sort(added.begin(),added.end());
 #ifdef LIBOX
 liboxns::LiBox<K,K> index(.5,.1,1);
 index.loadConfigByFile(config);
 auto insert=[&](K k){return index.insertKeyValue(k,k).status==liboxns::InsertStatus::SUCCESS;};
 auto lookup=[&](K k){auto r=index.searchKey(k);return r.status==liboxns::SearchStatus::SUCCESS&&r.value==k;};
 for(K k:initial)if(!insert(k)){std::cerr<<"INITIAL_INSERT_FAIL "<<k<<"\n";return 2;}
 #else
 ribtree::BTree<K,K> index(1,.5,.1);
 std::vector<std::pair<K,K>> data;for(K k:initial)data.emplace_back(k,k);
 index.bulkLoad(config,data);
 auto insert=[&](K k){index.insert(k,k);
#ifdef SCREEN_VALIDATE_TREE
 std::function<void(ribtree::NodeBase*)> visit=[&](auto* node){
 if(node->typeVersionLockObsolete.load() & 2){std::cerr<<"LOCKED_AFTER_INSERT,key="<<k<<",version="<<node->typeVersionLockObsolete.load()<<",type="<<int(node->type)<<"\n";throw std::runtime_error("Unreleased node lock");}
 if(node->type==ribtree::PageType::BTreeInner){auto* inner=static_cast<ribtree::BTreeInner<K>*>(node);for(unsigned i=0;i<=inner->count;i++)visit(inner->children[i]);}
 };visit(index.root.load());
#endif
 return true;};
 auto lookup=[&](K k){K v=-1;return index.lookup(k,v)&&v==k;};
 #endif
 std::vector<K> all=initial;
 auto check=[&](int phase){
 size_t missing=0;for(K k:all)if(!lookup(k)){if(missing<16){std::cerr<<"MISSING,"<<k<<"\n";

}missing++;}
 std::cerr<<"CHECK,"<<mode<<","<<phase<<","<<all.size()<<","<<missing<<"\n";
 return missing==0;
 };
 if(!check(0))return 3;
 for(int phase=0;phase<=4;phase++){
 if(phase){auto t=Clock::now();for(size_t i=(phase-1)*n/4;i<size_t(phase*n/4);i++){
 if(!insert(added[i])){std::cerr<<"INSERT_FAIL "<<added[i]<<"\n";return 4;}
 all.push_back(added[i]);}
 double secs=std::chrono::duration<double>(Clock::now()-t).count();
 std::cerr<<"INSERT,"<<mode<<","<<phase<<","<<(n/4)/secs/1e6<<"\n";
 if(!check(phase))return 5;
 }
 auto query=all;std::shuffle(query.begin(),query.end(),rng);
 size_t hits=0;auto t=Clock::now();for(int repeat=0;repeat<8;repeat++)for(K k:query)hits+=lookup(k);
 double secs=std::chrono::duration<double>(Clock::now()-t).count();
 std::cerr<<"LOOKUP,"<<mode<<","<<phase<<","<<query.size()*8/secs/1e6<<","<<hits<<"\n";
 }
 #ifdef LIBOX
 index.printAccumulatedSplitStats();
 #else
 index.printTreeStructure(config+".tree.txt");
 #endif
}
