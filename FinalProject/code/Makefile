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
SRCS = $(SRC_DIR)/test_dataset.cpp $(SRC_DIR)/knn.cu

# Target executable
TARGET = $(BUILD_DIR)/knn_test

# Link the target executable
$(TARGET):
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $(SRCS)

# Default target
all: clean $(TARGET)

# Run the target executable
run: $(TARGET)
	$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run