# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -O3

# Target executable
TARGET = build/compute_pi

# Source files
SRCS = src/compute_pi.cpp

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