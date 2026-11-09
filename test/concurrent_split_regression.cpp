#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <omp.h>
#ifdef LIBOX
#include "libox.h"
size_t MAX_KEYS_IN_SEGMENT = INT64_MAX;
#else
#include "btreeolc.h"
#endif
using Key = int64_t;

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::stoi(argv[1]) : 4;
    const uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 42;
    const bool ordered = argc > 3 && std::string(argv[3]) == "sorted";
    const Key count = 131072, domain = count * 4096;
    std::vector<Key> initial, added;
    std::unordered_set<Key> seen;
    for (Key i = 0; i < count; ++i) { initial.push_back(i * 4096); seen.insert(i * 4096); }
    std::mt19937_64 rng(seed);
    while (added.size() < size_t(count)) {
        Key key = domain / 2 + rng() % (domain / 100);
        if (seen.insert(key).second) added.push_back(key);
    }
    if (ordered) std::sort(added.begin(), added.end());
    const std::string config = "/tmp/index-concurrent-split-" + std::to_string(getpid()) + ".csv";
    struct RemoveFile { std::string path; ~RemoveFile() { std::remove(path.c_str()); } } cleanup{config};
    std::ofstream output(config);
    for (Key i = 0; i < 8; ++i) {
        output << i * domain / 8 << ',' << (i + 1) * domain / 8 << ',' << 4096 * 48;
#ifndef LIBOX
        output << ',' << count / 8;
#endif
        output << '\n';
    }
    output.close();
    omp_set_dynamic(0); omp_set_num_threads(threads);
#ifdef LIBOX
    liboxns::LiBox<Key, Key> index(.5, .1, threads);
    index.loadConfigByFile(config);
    auto put = [&](Key key) { return index.insertKeyValue(key, key ^ 0x5a5a5a5a).status == liboxns::InsertStatus::SUCCESS; };
    auto get = [&](Key key) { auto r = index.searchKey(key); return r.status == liboxns::SearchStatus::SUCCESS && r.value == (key ^ 0x5a5a5a5a); };
    for (Key key : initial) if (!put(key)) throw std::runtime_error("initial insertion failed");
#else
    ribtree::BTree<Key, Key> index(threads, .5, .1);
    std::vector<std::pair<Key, Key>> pairs;
    for (Key key : initial) pairs.emplace_back(key, key ^ 0x5a5a5a5a);
    index.bulkLoad(config, pairs);
    auto put = [&](Key key) { index.insert(key, key ^ 0x5a5a5a5a); return true; };
    auto get = [&](Key key) { Key value; return index.lookup(key, value) && value == (key ^ 0x5a5a5a5a); };
#endif
    for (unsigned phase = 1; phase <= 4; ++phase) {
        const size_t begin = (phase - 1) * added.size() / 4, end = phase * added.size() / 4;
        unsigned errors = 0;
#pragma omp parallel for reduction(+:errors) schedule(static)
        for (size_t i = begin; i < end; ++i) {
            errors += !put(added[i]);
            errors += !get(added[i]); // The just-inserted key must be visible.
            errors += !get(initial[i % initial.size()]);
        }
        for (Key key : initial) errors += !get(key);
        for (size_t i = 0; i < end; ++i) errors += !get(added[i]);
#ifndef LIBOX
        int leaf_depth = -1;
        std::function<Key(ribtree::NodeBase*, int)> check_tree = [&](auto* node, int depth) -> Key {
            if (!node || (node->typeVersionLockObsolete.load() & 2))
                throw std::runtime_error("invalid or locked tree node after split");
            if (node->type == ribtree::PageType::BTreeLeaf) {
                auto* leaf = static_cast<ribtree::BTreeLeaf<Key, Key>*>(node);
                if (leaf_depth < 0) leaf_depth = depth;
                if (depth != leaf_depth || !leaf->count || leaf->count > leaf->maxEntries)
                    throw std::runtime_error("invalid leaf depth or capacity");
                for (unsigned i = 0; i < leaf->count; ++i) {
                    if (leaf->keys[i] != leaf->segments[i]->getLowerBound() ||
                        (i && leaf->segments[i-1]->getUpperBound() != leaf->keys[i]))
                        throw std::runtime_error("invalid range descriptor ordering");
                }
                return leaf->keys[0];
            }
            auto* inner = static_cast<ribtree::BTreeInner<Key>*>(node);
            if (inner->count >= inner->maxEntries) throw std::runtime_error("invalid inner capacity");
            Key first = check_tree(inner->children[0], depth + 1);
            for (unsigned i = 0; i < inner->count; ++i)
                if (inner->keys[i] != check_tree(inner->children[i+1], depth + 1))
                    throw std::runtime_error("invalid separator after split");
            return first;
        };
        check_tree(index.root.load(), 0);
        if (ordered && phase == 4 && SPLIT_THRESHOLD <= 8 && leaf_depth < 2)
            throw std::runtime_error("deep split regression did not grow inner levels");
#endif
        if (errors) throw std::runtime_error("concurrent split lost or corrupted keys");
        std::cerr << "PASS phase=" << phase << " threads=" << threads << '\n';
    }
    // The hotspot contains more new keys than the original buckets can hold;
    // completing all inserts therefore necessarily exercises range rebuilding.
    std::cerr << "PASS concurrent split regression\n";
}
