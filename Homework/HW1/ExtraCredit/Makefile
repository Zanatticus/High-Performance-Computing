# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -fopenmp

# Target executable
TARGET = build/mat-mat-mul-openMP

# Source files
SRCS = src/mat-mat-mul-openMP.c

# Default target
all: clean $(TARGET) run

# Link the target executable
$(TARGET):
	$(CC) $(CFLAGS) -o $@ $(SRCS)

# Run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run