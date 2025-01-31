# Compiler
CXX = g++

# Compiler flags. The first commented line is for the baseline performance. The second line is the optimized performance.
CXXFLAGS = -pthread -std=c++11

# Target executable
TARGET = build/merge_sort

# Source files
SRCS = src/merge_sort.cpp

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