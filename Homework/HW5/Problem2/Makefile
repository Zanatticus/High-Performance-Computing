#!/bin/bash

.DEFAULT_GOAL := all

ROOT_DIR=$(PWD)
SRC_DIR=$(ROOT_DIR)/src
BUILD_DIR=$(ROOT_DIR)/build
OBJECT_NAME=cuda_tiling

# NVIDIA Tesla P100 Streaming Multiprocessor Architecture: sm_60
P100_ARCH=-arch sm_60
# NVIDIA Tesla V100 Streaming Multiprocessor Architecture: sm_70
V100_ARCH=-arch sm_70

ARCH=$(P100_ARCH)

# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -O3

# Source files
SRCS =  $(SRC_DIR)/$(OBJECT_NAME).cu

# Target executable
TARGET = $(BUILD_DIR)/$(OBJECT_NAME)

# Default target
all: clean $(TARGET) run

# Link the target executable
$(TARGET):
	module load nvidia-hpc-sdk && $(CC) $(ARCH) $(CFLAGS) -o $@ $(SRCS)

# Run the job using Slurm
run: $(TARGET)
	$(TARGET)
	
# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean run