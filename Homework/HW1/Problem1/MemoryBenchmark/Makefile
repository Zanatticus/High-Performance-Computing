# Compiler
CXX = g++

# Compiler flags. The first commented line is for the baseline performance. The second line is the optimized performance.
# CXXFLAGS = -std=c++11
CXXFLAGS = -std=c++11 -O3 -funroll-loops -march=native

# Target executable
TARGET = build/memory_benchmark

# Source files
SRCS = src/memory_benchmark.cpp

# Default target
all: clean $(TARGET) run

# Link the target executable
$(TARGET):
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

# Run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run