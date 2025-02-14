# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -fopenmp

# Target executable
TARGET = build/graph_coloring

# Source files
SRCS = src/graph_coloring.cpp

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