# Compiler
CC = g++

# Compiler flags
CFLAGS = -pthread

# Target executable
TARGET = merge_sort

# Source file
SRC = merge_sort.cpp

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
