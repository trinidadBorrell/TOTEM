#!/bin/bash

# Shell script to convert NPY reconstructed data to FIF format for TOTEM zero-shot learning
# Now handles session-based folder structure where each subject contains ses-* folders
# Input structure: sub-{ID}/ses-{num}/original.npy and reverted.npy
# Output structure: sub-{ID}/ses-{num}/*.fif files
#
# Provides 3 different ways to run the conversion:
# 1. Process all subjects in the folder
# 2. Process a specified number of random subjects
# 3. Process specific subjects by ID
#
# The script processes each session individually and creates separate FIF files.

# Example of usage:
# 1) Process all subjects
# ./step3_to_fif.sh all
# 2) Process 5 random subjects  
# ./step3_to_fif.sh random 5
# 3) Process specific subjects
# ./step3_to_fif.sh specific AA048 AA069 LP275
# Interactive mode
# ./step3_to_fif.sh interactive

# Configuration
NPY_DATA_BASE="/data/project/eeg_foundation/data/zero_shot_data/pydata"
ORIGINAL_DATA_BASE="/data/project/eeg_foundation/data/nice_derivatives/CONTROL_BIDS/nice_epochs_sfreq-100Hz_recombine-biosemi64"
OUTPUT_DIR="/data/project/eeg_foundation/data/zero_shot_data/fifdata"
SCRIPT_PATH="/home/triniborrell/home/projects/TOTEM/save_data/from_npy_to_fif_sessions.py"
SCALING_FACTOR="1e-5"  # Factor to scale reconstructed data to match original magnitude

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
    print_info "Checking required paths..."
    
    if [ ! -d "$NPY_DATA_BASE" ]; then
        print_error "NPY data base directory not found: $NPY_DATA_BASE"
        return 1
    fi
    
    if [ ! -d "$ORIGINAL_DATA_BASE" ]; then
        print_error "Original data base directory not found: $ORIGINAL_DATA_BASE"
        return 1
    fi
    
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_error "Python script not found: $SCRIPT_PATH"
        return 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    print_success "All paths validated"
    return 0
}

# Function to get available subjects (with session-based structure)
get_available_subjects() {
    local subjects=()
    if [ -d "$NPY_DATA_BASE" ]; then
        for dir in "$NPY_DATA_BASE"/sub-*; do
            if [ -d "$dir" ]; then
                local subject_id=$(basename "$dir" | sed 's/sub-//')
                # Check if this subject has any session folders with required files
                local has_sessions=false
                for session_dir in "$dir"/ses-*; do
                    if [ -d "$session_dir" ] && [ -f "$session_dir/original.npy" ] && [ -f "$session_dir/reverted.npy" ]; then
                        has_sessions=true
                        break
                    fi
                done
                if [ "$has_sessions" = true ]; then
                    subjects+=("$subject_id")
                fi
            fi
        done
    fi
    echo "${subjects[@]}"
}

# Function to display help
show_help() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Convert NPY reconstructed data to FIF format"
    echo ""
    echo "Modes:"
    echo "  all                     Process all available subjects"
    echo "  random N                Process N random subjects"
    echo "  specific SUB1 [SUB2...] Process specific subjects by ID"
    echo "  interactive             Interactive mode to select subjects"
    echo "  help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all"
    echo "  $0 random 5"
    echo "  $0 specific AA048 LP275"
    echo "  $0 interactive"
    echo ""
    echo "Configuration:"
    echo "  NPY data base: $NPY_DATA_BASE"
    echo "  Original data: $ORIGINAL_DATA_BASE"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Python script: $SCRIPT_PATH"
    echo "  Scaling factor: $SCALING_FACTOR"
}

# Function to process a single subject's sessions
process_subject_sessions() {
    local subject_id="$1"
    local subject_dir="$NPY_DATA_BASE/sub-$subject_id"
    
    print_info "Processing sessions for subject: $subject_id"
    
    if [ ! -d "$subject_dir" ]; then
        print_error "Subject directory not found: $subject_dir"
        return 1
    fi
    
    # Use the new simplified Python script approach - it will auto-discover and process all sessions
    print_info "Converting all sessions for subject $subject_id to FIF format..."
    
    python3 "$SCRIPT_PATH" \
        --subject_id "$subject_id" \
        --npy_data_base "$NPY_DATA_BASE" \
        --output_dir "$OUTPUT_DIR" \
        --original_data_dir "$ORIGINAL_DATA_BASE" \
        --scaling_factor "$SCALING_FACTOR"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully converted sessions for subject: $subject_id"
        return 0
    else
        print_error "Failed to convert sessions for subject: $subject_id"
        return 1
    fi
}

# Function to run the conversion for multiple subjects
run_conversion() {
    local mode="$1"
    shift
    local subjects=()
    
    case "$mode" in
        "all")
            subjects=($(get_available_subjects))
            if [ ${#subjects[@]} -eq 0 ]; then
                print_error "No subjects with session data found"
                return 1
            fi
            print_info "Processing all ${#subjects[@]} subjects..."
            ;;
        "specific")
            subjects=("$@")
            print_info "Processing specific subjects: ${subjects[*]}"
            ;;
        "random")
            local num_subjects="$1"
            local available_subjects=($(get_available_subjects))
            if [ ${#available_subjects[@]} -eq 0 ]; then
                print_error "No subjects with session data found"
                return 1
            fi
            if [ "$num_subjects" -gt "${#available_subjects[@]}" ]; then
                print_warning "Requested $num_subjects subjects, but only ${#available_subjects[@]} available. Processing all."
                subjects=("${available_subjects[@]}")
            else
                # Randomly select subjects
                subjects=($(printf '%s\n' "${available_subjects[@]}" | shuf | head -n "$num_subjects"))
            fi
            print_info "Processing $num_subjects random subjects: ${subjects[*]}"
            ;;
        *)
            print_error "Unknown conversion mode: $mode"
            return 1
            ;;
    esac
    
    # Process each subject
    local success_count=0
    local failure_count=0
    local total_sessions_processed=0
    
    for subject_id in "${subjects[@]}"; do
        echo ""
        echo "========================================"
        if process_subject_sessions "$subject_id"; then
            ((success_count++))
            # Count sessions for this subject
            local session_count=$(find "$NPY_DATA_BASE/sub-$subject_id" -maxdepth 1 -type d -name "ses-*" | wc -l)
            ((total_sessions_processed += session_count))
        else
            ((failure_count++))
        fi
        echo "========================================"
    done
    
    echo ""
    print_info "Conversion complete!"
    print_success "Successfully processed: $success_count subjects"
    print_success "Total sessions processed: $total_sessions_processed"
    if [ $failure_count -gt 0 ]; then
        print_error "Failed to process: $failure_count subjects"
    fi
    
    return 0
}

# Function for interactive mode
interactive_mode() {
    print_info "Interactive mode - Select subjects to process"
    echo ""
    
    local available_subjects=($(get_available_subjects))
    
    if [ ${#available_subjects[@]} -eq 0 ]; then
        print_error "No subjects with session data found in $NPY_DATA_BASE"
        return 1
    fi
    
    print_info "Available subjects (${#available_subjects[@]} total):"
    for i in "${!available_subjects[@]}"; do
        echo "  $((i+1)). ${available_subjects[$i]}"
    done
    echo ""
    
    echo "Options:"
    echo "  a) Process all subjects"
    echo "  r) Process random number of subjects"
    echo "  s) Select specific subjects"
    echo "  q) Quit"
    echo ""
    
    read -p "Choose an option [a/r/s/q]: " choice
    
    case $choice in
        a|A)
            print_info "Processing all subjects..."
            run_conversion "all"
            ;;
        r|R)
            read -p "Enter number of random subjects to process: " num_random
            if [[ "$num_random" =~ ^[0-9]+$ ]] && [ "$num_random" -gt 0 ]; then
                print_info "Processing $num_random random subjects..."
                run_conversion "random" "$num_random"
            else
                print_error "Invalid number: $num_random"
                return 1
            fi
            ;;
        s|S)
            echo "Enter subject IDs separated by spaces (e.g., AA048 LP275):"
            read -p "Subjects: " -a selected_subjects
            if [ ${#selected_subjects[@]} -gt 0 ]; then
                print_info "Processing selected subjects: ${selected_subjects[*]}"
                run_conversion "specific" "${selected_subjects[@]}"
            else
                print_error "No subjects specified"
                return 1
            fi
            ;;
        q|Q)
            print_info "Exiting..."
            return 0
            ;;
        *)
            print_error "Invalid option: $choice"
            return 1
            ;;
    esac
}

# Main script logic
main() {
    print_info "TOTEM Zero-shot NPY to FIF Converter"
    print_info "======================================"
    echo ""
    
    # Check if paths exist
    if ! check_paths; then
        exit 1
    fi
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    local mode="$1"
    shift
    
    case "$mode" in
        "all")
            print_info "Processing all available subjects..."
            run_conversion "all"
            ;;
        "random")
            if [ $# -eq 0 ]; then
                print_error "Random mode requires number of subjects"
                echo "Usage: $0 random N"
                exit 1
            fi
            local num_subjects="$1"
            if [[ "$num_subjects" =~ ^[0-9]+$ ]] && [ "$num_subjects" -gt 0 ]; then
                print_info "Processing $num_subjects random subjects..."
                run_conversion "random" "$num_subjects"
            else
                print_error "Invalid number of subjects: $num_subjects"
                exit 1
            fi
            ;;
        "specific")
            if [ $# -eq 0 ]; then
                print_error "Specific mode requires at least one subject ID"
                echo "Usage: $0 specific SUB1 [SUB2 ...]"
                exit 1
            fi
            local subjects=("$@")
            print_info "Processing specific subjects: ${subjects[*]}"
            run_conversion "specific" "${subjects[@]}"
            ;;
        "interactive")
            interactive_mode
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
