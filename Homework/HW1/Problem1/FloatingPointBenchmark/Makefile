# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -O2

# Target executable
TARGET = build/floating_point_benchmark

# Source files
SRCS = src/floating_point_benchmark.cpp

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