# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -fopenmp -std=c++11

# Source directory
SRC_DIR = src

# Build directory
BUILD_DIR = build

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%, $(SRCS))

# Default target
all: clean $(OBJS)

# Link the target executable
$(BUILD_DIR)/%: $(SRC_DIR)/%.c
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean up build files
clean:
	rm -f $(BUILD_DIR)/*

.PHONY: all clean run