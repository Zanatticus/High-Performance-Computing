#!/bin/bash

###########################
# ------ CONSTANTS ------ #
###########################

.DEFAULT_GOAL := all

ROOT_DIR := $(PWD)
SRC_DIR := $(ROOT_DIR)/src
INCLUDE_DIR := $(ROOT_DIR)/include
BUILD_DIR := $(ROOT_DIR)/build

# NVIDIA Tesla P100 Streaming Multiprocessor Architecture: sm_60
P100_ARCH=-arch sm_60
# NVIDIA Tesla V100 Streaming Multiprocessor Architecture: sm_70
V100_ARCH=-arch sm_70

ARCH=$(P100_ARCH)

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
CPP_SRCS := $(SRC_DIR)/MNIST-Loader.cpp \
			$(SRC_DIR)/CIFAR-Loader.cpp
CU_SRCS := $(SRC_DIR)/main.cu \
			$(SRC_DIR)/KNN-Classifier.cu

# Target executable
TARGET := $(BUILD_DIR)/main

# Compile C++ sources, CUDA sources, and link the target executable
$(TARGET):
	module load nvidia-hpc-sdk && $(NVCC) $(NVCCFLAGS) -o $(TARGET) $(ARCH) $(CU_SRCS) $(CPP_SRCS)

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