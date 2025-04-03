#!/bin/bash

###########################
# ------ CONSTANTS ------ #
###########################

.DEFAULT_GOAL := all

ROOT_DIR := $(PWD)
SRC_DIR := $(ROOT_DIR)/src
INCLUDE_DIR := $(ROOT_DIR)/include
BUILD_DIR := $(ROOT_DIR)/build

###########################
# ----- COMPILATION ----- #
###########################

# Compilers
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -O3 -I$(INCLUDE_DIR)
NVCCFLAGS := -O3 -I$(INCLUDE_DIR)

# Source files
CPP_SRCS := $(SRC_DIR)/MNIST-Loader.cpp
CU_SRCS := $(SRC_DIR)/main.cu

# Target executable
TARGET := $(BUILD_DIR)/main

# Compile C++ sources, CUDA sources, and link the target executable
$(TARGET):
	module load nvidia-hpc-sdk && $(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CU_SRCS) $(CPP_SRCS)

###########################
# -------- RULES -------- #
###########################

# Default target
all: clean $(TARGET)

# Run the target executable
run: $(TARGET)
	$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean