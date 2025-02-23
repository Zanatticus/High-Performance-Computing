# Compiler
CC = gcc

# Compiler flags
CFLAGS =

# Linker flags
LDFLAGS = -lopenblas

# Target executable
TARGET = build/blas_simple

# Source files
SRCS = src/blas_simple.c

# Default target
all: clean $(TARGET)

# Link the target executable
$(TARGET):
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# Run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run