# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O3

# Target executable
TARGET = build/parallel_histogram

# Source files
SRCS = src/parallel_histogram.cpp

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