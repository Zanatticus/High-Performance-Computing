# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -O3

# Source directory
SRC_DIR = src

# Build directory
BUILD_DIR = build

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%, $(SRCS))

# Default target
all: clean $(OBJS)

# Link the target executable
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	module load nvidia-hpc-sdk && $(CC) $(CFLAGS) -o $@ $<

# Clean up build files
clean:
	rm -f $(BUILD_DIR)/*

.PHONY: all clean