#!/bin/bash

# Shell script to process EEG data for TOTEM zero-shot learning
# Provides 3 different ways to run the processing:
# 1. Process all subjects in the folder
# 2. Process a specified number of random subjects
# 3. Process specific subjects by ID


#Example of usage:
# 1) Process all subjects
# ./step1_process_eeg_epochs.sh all
# 2) Process 5 random subjects  
# ./step1_process_eeg_epochs.sh random 5
# 3) Process specific subjects
# ./step1_process_eeg_epochs.sh specific PD155 LP275 AA069
# Interactive mode
# ./step1_process_eeg_epochs.sh interactive

# Configuration
CONTROL_BIDS=false
if [ "$CONTROL_BIDS" = true ]; then
    BASE_PATH="/data/project/eeg_foundation/data/nice_derivatives/CONTROL_BIDS/nice_epochs_sfreq-100Hz_recombine-biosemi64"
else
    BASE_PATH="/data/project/eeg_foundation/data/data_250Hz_EGI256/nice_epochs4"
fi
SAVE_PATH="/data/project/eeg_foundation/data/data_250Hz_EGI256/processed_nice_data_256/DOC"
SCRIPT_PATH="$(dirname "$(realpath "$0")")/../process_zero_shot_data/process_eeg_data_zero_shot.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required paths exist
check_paths() {
    if [ ! -d "$BASE_PATH" ]; then
        print_error "Base path does not exist: $BASE_PATH"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_error "Python script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    # Create save directory if it doesn't exist
    mkdir -p "$SAVE_PATH"
    print_info "Save path: $SAVE_PATH"
}

# Function to get available subjects
get_subjects() {
    find "$BASE_PATH" -maxdepth 1 -type d -name "sub-*" | sort
}

# Function to count available subjects
count_subjects() {
    get_subjects | wc -l
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "MODES:"
    echo "  all                    Process all subjects in the folder"
    echo "  random [N]             Process N random subjects (default: 3)"
    echo "  specific [SUB1 SUB2...]  Process specific subjects by ID"
    echo "  interactive            Interactive mode (default)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 all"
    echo "  $0 random 5"
    echo "  $0 specific PD155 LP275 AA069"
    echo "  $0 interactive"
    echo ""
}

# Function for interactive mode
interactive_mode() {
    local total_subjects=$(count_subjects)
    
    echo ""
    echo "========================================"
    echo "  EEG Data Processing - Interactive Mode"
    echo "========================================"
    echo ""
    print_info "Found $total_subjects subjects in $BASE_PATH"
    echo ""
    echo "Choose processing mode:"
    echo "1) Process ALL subjects ($total_subjects subjects)"
    echo "2) Process a specific NUMBER of random subjects"
    echo "3) Process SPECIFIC subjects by ID"
    echo "4) Show available subjects and exit"
    echo "5) Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            process_all_subjects
            ;;
        2)
            read -p "Enter number of random subjects to process: " num_subjects
            if [[ $num_subjects =~ ^[0-9]+$ ]] && [ $num_subjects -gt 0 ]; then
                process_random_subjects $num_subjects
            else
                print_error "Invalid number. Please enter a positive integer."
                exit 1
            fi
            ;;
        3)
            echo "Available subjects:"
            get_subjects | sed 's|.*/||' | sed 's/sub-//' | sort | paste -d' ' - - - - -
            echo ""
            read -p "Enter subject IDs (space-separated, e.g., PD155 LP275 AA069): " -a subject_ids
            if [ ${#subject_ids[@]} -gt 0 ]; then
                process_specific_subjects "${subject_ids[@]}"
            else
                print_error "No subjects specified."
                exit 1
            fi
            ;;
        4)
            echo ""
            print_info "Available subjects:"
            get_subjects | sed 's|.*/||' | sort
            exit 0
            ;;
        5)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please enter 1-5."
            exit 1
            ;;
    esac
}

# Function to process all subjects
process_all_subjects() {
    local total_subjects=$(count_subjects)
    print_info "Processing ALL $total_subjects subjects..."
    
    python3 "$SCRIPT_PATH" \
        --base_path "$BASE_PATH" \
        --save_path "$SAVE_PATH" \
        --n_subjects $total_subjects
}

# Function to process random subjects
process_random_subjects() {
    local num_subjects=$1
    print_info "Processing $num_subjects random subjects..."
    
    python3 "$SCRIPT_PATH" \
        --base_path "$BASE_PATH" \
        --save_path "$SAVE_PATH" \
        --n_subjects $num_subjects \
        --random
}

# Function to process specific subjects
process_specific_subjects() {
    local subjects=("$@")
    print_info "Processing specific subjects: ${subjects[*]}"
    
    python3 "$SCRIPT_PATH" \
        --base_path "$BASE_PATH" \
        --save_path "$SAVE_PATH" \
        --subjects "${subjects[@]}"
}

# Main script logic
main() {
    print_info "EEG Data Processing Script"
    print_info "Base path: $BASE_PATH"
    
    # Check if paths exist
    check_paths
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        # No arguments - run interactive mode
        interactive_mode
    elif [ "$1" = "all" ]; then
        process_all_subjects
    elif [ "$1" = "random" ]; then
        local num_subjects=${2:-3}  # Default to 3 if not specified
        if [[ $num_subjects =~ ^[0-9]+$ ]] && [ $num_subjects -gt 0 ]; then
            process_random_subjects $num_subjects
        else
            print_error "Invalid number for random mode: $num_subjects"
            show_usage
            exit 1
        fi
    elif [ "$1" = "specific" ]; then
        if [ $# -lt 2 ]; then
            print_error "No subjects specified for specific mode."
            show_usage
            exit 1
        fi
        shift  # Remove 'specific' from arguments
        process_specific_subjects "$@"
    elif [ "$1" = "interactive" ]; then
        interactive_mode
    elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    else
        print_error "Unknown mode: $1"
        show_usage
        exit 1
    fi
}

# Run main function
main "$@"
