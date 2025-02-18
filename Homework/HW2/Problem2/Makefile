# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -pthread -O3

# Target executable
TARGET = build/dining_philosophers

# Source files
SRCS = src/dining_philosophers.cpp

# Default target
all: clean $(TARGET)

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