#!/bin/bash

# Define the file extensions to look for
extensions=("*.c" "*.cpp" "*.h" "*.hpp")

# Find and format files
for ext in "${extensions[@]}"; do
    find . -type f -name "$ext" -exec ./clang-format -i {} \;
done
