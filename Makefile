CXX = g++

# Source files
BENCHMARK_SRC = ./test/benchmark.cpp
PARTITION_SRC = ./src/partition_optimization.cpp
SEGMENT_SRC = ./src/segmentation.cpp
HEADERS = ./src/libox.h ./src/segmentation.h ./src/libox_utils.h

# Targets
BENCHMARK = ./test/benchmark
BENCHMARK_DEBUG = ./test/benchmark_debug
BENCHMARK_PROF = ./test/benchmark_prof
PARTITION = ./test/partition_optimization
PARTITION_DEBUG = ./test/partition_optimization_debug
SEGMENT = segmentation
SEGMENT_DEBUG = segmentation_debug

# Flags
COMMON_FLAGS = --std=c++20 -faligned-new -march=native -fopenmp
RELEASE_FLAGS = -O3 -DNDEBUG
DEBUG_FLAGS = -O0 -g
PROF_FLAGS = -O2 -pg -DNDEBUG
STD_FLAGS = --std=c++20

.PHONY: all debug prof partition partition_debug segment segment_debug clean

all: $(BENCHMARK) $(PARTITION) $(PARTITION_DEBUG) $(BENCHMARK_DEBUG)

# Benchmark
$(BENCHMARK): $(HEADERS) $(BENCHMARK_SRC)
	$(CXX) $(RELEASE_FLAGS) $(COMMON_FLAGS) $(BENCHMARK_SRC) -o $(BENCHMARK)

debug: $(BENCHMARK_DEBUG)
$(BENCHMARK_DEBUG): $(HEADERS) $(BENCHMARK_SRC)
	$(CXX) $(DEBUG_FLAGS) $(COMMON_FLAGS) $(BENCHMARK_SRC) -o $(BENCHMARK_DEBUG)

prof: $(BENCHMARK_PROF)
$(BENCHMARK_PROF): $(HEADERS) $(BENCHMARK_SRC)
	$(CXX) $(PROF_FLAGS) $(COMMON_FLAGS) $(BENCHMARK_SRC) -o $(BENCHMARK_PROF)

# Partition
partition: $(PARTITION)
$(PARTITION): $(PARTITION_SRC)
	$(CXX) $(STD_FLAGS) $(RELEASE_FLAGS) $(PARTITION_SRC) -o $(PARTITION)

partition_debug: $(PARTITION_DEBUG)
$(PARTITION_DEBUG): $(PARTITION_SRC)
	$(CXX) $(STD_FLAGS) $(DEBUG_FLAGS) $(PARTITION_SRC) -o $(PARTITION_DEBUG)

# Segment
segment: $(SEGMENT)
$(SEGMENT): $(SEGMENT_SRC)
	$(CXX) $(STD_FLAGS) $(RELEASE_FLAGS) $(SEGMENT_SRC) -o $(SEGMENT)

segment_debug: $(SEGMENT_DEBUG)
$(SEGMENT_DEBUG): $(SEGMENT_SRC)
	$(CXX) $(STD_FLAGS) $(DEBUG_FLAGS) $(SEGMENT_SRC) -o $(SEGMENT_DEBUG)

clean:
	rm -f $(BENCHMARK) $(BENCHMARK_DEBUG) $(BENCHMARK_PROF) \
	      $(PARTITION) $(PARTITION_DEBUG) $(SEGMENT) $(SEGMENT_DEBUG)
