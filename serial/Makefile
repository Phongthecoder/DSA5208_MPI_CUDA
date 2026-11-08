CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
BUILD_DIR ?= build

.PHONY: all check benchmark clean

all: $(BUILD_DIR)/serial_pf

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/particle_filter.o: src/particle_filter.cpp src/particle_filter.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -c $< -o $@

$(BUILD_DIR)/serial_pf: src/serial.cpp $(BUILD_DIR)/particle_filter.o src/particle_filter.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(LDFLAGS) $< $(BUILD_DIR)/particle_filter.o $(LDLIBS) -o $@

$(BUILD_DIR)/validate_serial: tests/validate_serial.cpp $(BUILD_DIR)/particle_filter.o src/particle_filter.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -Isrc $(LDFLAGS) $< $(BUILD_DIR)/particle_filter.o $(LDLIBS) -o $@

check: $(BUILD_DIR)/validate_serial
	$(BUILD_DIR)/validate_serial

benchmark: $(BUILD_DIR)/serial_pf
	python3 scripts/benchmark_serial.py --binary $(BUILD_DIR)/serial_pf

clean:
	rm -rf -- $(BUILD_DIR)
