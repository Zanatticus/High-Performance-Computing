# Compiler
CC = gcc

# Compiler flags
CFLAGS = -fopenmp

# Target executable
TARGET = build/mat-mat-mul-openMP

# Source files
SRCS = src/mat-mat-mul-openMP.c

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