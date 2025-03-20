# Compiler
CXX = mpic++

# Compiler flags
CXXFLAGS = -O3

# Target executable
TARGET = build/MPI_counter

# Source files
SRCS = src/MPI_counter.cpp

# Default target
all: clean $(TARGET)

# Link the target executable
$(TARGET):
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean