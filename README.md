# LiBox

A concurrent, in-memory numerical index using an interpolation directory,
range-organized segments and SIMD fingerprint filtering inside boxes.
The interpolation table retains 4-byte segment indices. The common lookup path
checks the indexed key interval before using the existing correction search.
Same-size directory replacements
repair only affected table cells; directory growth or outer-bound changes rebuild
the grid at its original density. Vector insertion can still move directory entries.
Table cells remain `int32_t`; the optimization does not enlarge the table.

## Requirements and build

- Linux, an x86-64 CPU with AVX-512 support, and OpenMP.
- A C++20 compiler with `std::atomic_ref` support.
- Python 3 for the regression runners.

```sh
make all
```

The benchmark is `test/benchmark`; the configuration tool is
`test/partition_optimization`.

## Basic use

```cpp
#include "src/libox.h"

size_t MAX_KEYS_IN_SEGMENT = INT64_MAX;

int main() {
    liboxns::LiBox<int64_t, int64_t> index(0.5, 0.1, 4);
    index.insertKeyValue(10, 100);  // No config file is needed for an empty index.
    index.insertKeyValue(10, 200);  // Overwrite the existing value.
    auto result = index.searchKey(10);
    auto removed = index.deleteKey(10);
    return result.status == liboxns::SearchStatus::SUCCESS &&
           result.value == 200 && removed.found ? 0 : 1;
}
```

Compile with `-std=c++20 -O3 -march=native -fopenmp -pthread -I. -Isrc`.

## Configuration and bulk loading

Configuration is optional for online integer insertion. `loadConfigByFile(path)`
accepts CSV rows of the form:

```text
lower,upper,box_range
```

Bounds are half-open `[lower, upper)`, box widths must be positive, and ranges
must be ordered, contiguous and nonoverlapping. Blank lines are ignored.
A fourth key-count field is accepted for compatibility. Invalid configurations
are rejected before replacing the current configuration.

`loadConfigFromData(keys)` sorts and deduplicates keys to generate ranges; it does
not insert values. `bulk_load(pairs, count)` inserts key/value pairs, and
`buildIndex(&keys)` inserts keys with value `1`. Repeated keys overwrite the
stored value; parallel writes to the same key have no input-order guarantee.

The small files under `sample/` support this runnable example:

```sh
./test/benchmark \
  --keys_file=sample/sample_keys.txt \
  --keys_file_type=text \
  --config_file_path=sample/sample_config.csv \
  --init_num_keys=1000 --total_num_keys=2000 \
  --batch_size=1000 --insert_frac=0.5 --thread_num=4
```

## Regression checks

```sh
python3 test/run_concurrency_checks.py
python3 test/run_extended_checks.py --tsan
```

Add `--quick` for fewer seeds. The runners need Linux `taskset`; TSan checks also
use `setarch -R`. They write local binaries and logs under `build/`. ASan/UBSan
checks enable leak detection. See the test sources for covered operations.

## API limits

- There is no public range-scan API.
- Online bootstrap/range growth requires integer keys. The largest representable
  integer key is rejected because its exclusive upper bound is not representable.
- Configure/bulk-load an index before concurrent use. Do not reconfigure or
  destroy an index while another thread is using it.
