MPICXX ?= mpicxx
NVCC ?= nvcc
CUDA_ARCH ?= 86
CUDA_HOME ?= $(shell dirname "$$(dirname "$$(command -v $(NVCC))")")
BUILD_DIR ?= build
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O3 --fmad=false -arch=sm_$(CUDA_ARCH)

CPPFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -I../mpi/src -I../serial/src -Isrc
MPI_COMPILE_FLAGS = $(patsubst -pthread,-Xcompiler=-pthread,$(shell $(MPICXX) --showme:compile))
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/cuda_filter.o $(BUILD_DIR)/mpi_filter.o $(BUILD_DIR)/serial_filter.o

.PHONY: all

all: $(BUILD_DIR)/cuda_pf

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main.o: src/main.cpp src/cuda_filter.hpp ../mpi/src/mpi_filter.hpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/cuda_filter.o: src/cuda_filter.cu src/cuda_filter.hpp ../mpi/src/philox.hpp ../mpi/src/mpi_filter.hpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(MPI_COMPILE_FLAGS) $(NVCCFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/mpi_filter.o: ../mpi/src/mpi_filter.cpp ../mpi/src/mpi_filter.hpp ../mpi/src/philox.hpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/serial_filter.o: ../serial/src/particle_filter.cpp ../serial/src/particle_filter.hpp | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/cuda_pf: $(OBJECTS)
	$(MPICXX) $(LDFLAGS) $(OBJECTS) -L"$(CUDA_HOME)/lib64" -Wl,-rpath,"$(CUDA_HOME)/lib64" -lcudart $(LDLIBS) -o $@
