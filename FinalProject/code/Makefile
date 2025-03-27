#!/bin/bash

ROOT_DIR=$(PWD)
SRC_DIR=$(ROOT_DIR)/src
INCLUDE_DIR=$(ROOT_DIR)/include
BUILD_DIR=$(ROOT_DIR)/build

# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -O3

# Source files
SRCS = $(SRC_DIR)/KNN.cu

# Target executable
TARGET = $(BUILD_DIR)/KNN

# Link the target executable
$(TARGET):
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $(SRCS)

# Default target
all: clean $(TARGET) run

# Run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run