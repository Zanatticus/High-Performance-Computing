# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -mavx512f

# Target executable
TARGET = build/mat_vec_mul_avx

# Source files
SRCS = src/mat_vec_mul_avx.c

# Default target
all: clean $(TARGET) run

# Compile the source files to assembly
assembly: $(SRCS)
	$(CC) $(CFLAGS) -S -o $(SRCS:.c=.s) $(SRCS)

# Link the target executable
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

# Run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET) src/*.s

.PHONY: all clean run assembly