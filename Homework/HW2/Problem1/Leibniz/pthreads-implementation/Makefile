# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -pthread

# Target executable
TARGET = build/leibniz_pthreads

# Source files
SRCS = src/leibniz_pthreads.cpp

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