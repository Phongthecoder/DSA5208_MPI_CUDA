MPICXX ?= mpicxx
MPIEXEC ?= mpiexec
MPIEXEC_FLAGS ?=
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
BUILD_DIR ?= build
CHECK_RANKS ?= 1 3 8

CPPFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -I../serial/src -Isrc
COMMON = $(BUILD_DIR)/mpi_filter.o $(BUILD_DIR)/serial_filter.o

.PHONY: all check clean

all: $(BUILD_DIR)/mpi_pf $(BUILD_DIR)/resampling_bench

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/mpi_filter.o: src/mpi_filter.cpp src/mpi_filter.hpp src/philox.hpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/serial_filter.o: ../serial/src/particle_filter.cpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/mpi_pf: src/main.cpp src/mpi_filter.hpp $(COMMON)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(LDFLAGS) $< $(COMMON) $(LDLIBS) -o $@

$(BUILD_DIR)/resampling_bench: src/resampling_bench.cpp src/mpi_filter.hpp src/philox.hpp $(COMMON)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(LDFLAGS) $< $(COMMON) $(LDLIBS) -o $@

$(BUILD_DIR)/validate_mpi: tests/validate_mpi.cpp src/mpi_filter.hpp src/philox.hpp $(COMMON)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(LDFLAGS) $< $(COMMON) $(LDLIBS) -o $@

check: $(BUILD_DIR)/validate_mpi
	set -e; for ranks in $(CHECK_RANKS); do $(MPIEXEC) $(MPIEXEC_FLAGS) -n $$ranks $<; done

clean:
	rm -rf -- $(BUILD_DIR)
