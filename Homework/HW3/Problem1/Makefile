# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11

# Target executable
TARGET = build/IEEE754_Computations

# Source files
SRCS = src/IEEE754_Computations.cpp

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