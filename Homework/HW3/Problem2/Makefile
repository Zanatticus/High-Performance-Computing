# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -mavx512f

# Target executable
TARGET = build/dot-product-AVX

# Source files
SRCS = src/dot-product-AVX.c

# Default target
all: clean $(TARGET)

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