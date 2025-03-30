#!/bin/bash

# Define the file extensions to look for
extensions=("*.c" "*.cpp" "*.h" "*.hpp" "*.cu")

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Format C/C++/CUDA files in specified directories"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -a, --all      Format files in all subdirectories (default)"
    echo "  -n, --name=DIR Format files only in the specified directory"
    echo ""
    echo "Example:"
    echo "  $0 --all"
    echo "  $0 --name=src"
    echo "  $0 -n src"
    exit 0
}

# Default to all directories
target_dir="."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -a|--all)
            target_dir="."
            shift
            ;;
        -n)
            if [ -z "$2" ]; then
                echo "Error: Directory name required for -n option"
                show_help
            fi
            target_dir="$2"
            shift 2
            ;;
        --name=*)
            target_dir="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Find and format files
for ext in "${extensions[@]}"; do
    find "$target_dir" -type f -name "$ext" -exec ./clang-format -i {} \;
done
