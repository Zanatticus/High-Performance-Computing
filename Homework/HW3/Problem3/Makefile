# Compiler
CC = gcc

# Compiler flags
CFLAGS = 

# Target executable
TARGET = build/matmul

# Source files
SRCS = src/matmul.c

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