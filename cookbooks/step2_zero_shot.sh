#!/bin/bash

# Shell script to extract VQVAE codes for zero-shot forecasting
# Now handles session-based folder structure where each subject contains ses-* folders
# Input structure: sub-{ID}/ses-{num}/data.npy and trial_labels.npy
# Output structure: sub-{ID}/ses-{num}/original.npy, codes.npy, reverted.npy, codebook.npy
# 
# Provides 3 different ways to run the processing:
# 1. Process all subjects in the folder
# 2. Process a specified number of random subjects
# 3. Process specific subjects by ID

# Example of usage:
# 1) Process all subjects
# ./step2_zero_shot.sh all
# 2) Process 5 random subjects  
# ./step2_zero_shot.sh random 5
# 3) Process specific subjects
# ./step2_zero_shot.sh specific sub-PD155 sub-LP275 sub-AA069
# Interactive mode
# ./step2_zero_shot.sh interactive

# Configuration
BASE_PATH="/data/project/eeg_foundation/data/data_250Hz_EGI256/processed_nice_data_256/DOC"
SAVE_BASE_PATH="/data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/pydata"
SCRIPT_PATH="$(dirname "$(realpath "$0")")/../forecasting/extract_zero_shot_data_single_df.py"
VQVAE_MODEL_PATH="/home/triniborrell/home/projects/TOTEM/forecasting/pretrained/forecasting/checkpoints/final_model.pth"

# Parameters
GPU=0
RANDOM_SEED=2021
DATA_NAME="neuro_zero_shot"
TYPE_PRETRAINED="forecast"
COMPRESSION_FACTOR=4

# Memory Management Settings
BATCH_SIZE=32  # Reduced from default 256 to prevent CUDA OOM
# If you still get OOM errors, try reducing to 16 or 8
# If you have more GPU memory, you can increase to 64 or 128

# Note: SEQ_LEN and PRED_LEN not needed for whole data processing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to clear GPU memory
clear_gpu_memory() {
    print_info "Clearing GPU memory..."
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0):,} bytes')
else:
    print('CUDA not available')
" 2>/dev/null || print_warning "Could not clear GPU memory"
}

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
    
    if [ ! -f "$VQVAE_MODEL_PATH" ]; then
        print_error "VQVAE model not found: $VQVAE_MODEL_PATH"
        exit 1
    fi
    
    # Create save directory if it doesn't exist
    mkdir -p "$SAVE_BASE_PATH"
    print_info "Save base path: $SAVE_BASE_PATH"
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
    echo "  random N               Process N random subjects"
    echo "  specific SUB1 SUB2...  Process specific subjects by ID"
    echo "  interactive            Interactive mode to select subjects"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 all"
    echo "  $0 random 5"
    echo "  $0 specific sub-PD155 sub-LP275 sub-AA069"
    echo "  $0 interactive"
    echo ""
}

# Function to process a single subject with sessions
process_subject() {
    local subject=$1
    local subject_id=$(basename "$subject" | sed 's/sub-//')
    
    print_info "Processing subject: $subject_id"
    
    # Check if subject data exists
    if [ ! -d "$subject" ]; then
        print_error "Subject directory not found: $subject"
        return 1
    fi
    
    # Find all session directories in the subject folder
    local session_dirs=($(find "$subject" -maxdepth 1 -type d -name "ses-*" | sort))
    
    if [ ${#session_dirs[@]} -eq 0 ]; then
        print_error "No session directories (ses-*) found in $subject"
        return 1
    fi
    
    print_info "Found ${#session_dirs[@]} sessions for subject $subject_id"
    
    local session_success_count=0
    local session_failure_count=0
    
    # Process each session
    for session_dir in "${session_dirs[@]}"; do
        local session_name=$(basename "$session_dir")
        print_info "Processing $session_name for subject $subject_id"
        
        # Check if data.npy exists in session
        if [ ! -f "$session_dir/data.npy" ]; then
            print_error "data.npy not found in $session_dir"
            ((session_failure_count++))
            continue
        fi
        
        # Check if trial_labels.npy exists in session
        if [ ! -f "$session_dir/trial_labels.npy" ]; then
            print_warning "trial_labels.npy not found in $session_dir (continuing anyway)"
        fi
        
        # Create session save directory
        session_save_path="$SAVE_BASE_PATH/sub-$subject_id/$session_name"
        mkdir -p "$session_save_path"
        
        print_info "Extracting VQVAE codes for subject $subject_id, $session_name..."
        
        # Set CUDA memory optimization environment variables
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export CUDA_LAUNCH_BLOCKING=1
        
        # Run the Python extraction script for this session
        python3 "$SCRIPT_PATH" \
            --data "$DATA_NAME" \
            --root_path "$session_dir/" \
            --data_path "$DATA_NAME" \
            --features M \
            --enc_in 64 \
            --gpu $GPU \
            --save_path "$session_save_path/" \
            --trained_vqvae_model_path "$VQVAE_MODEL_PATH" \
            --compression_factor $COMPRESSION_FACTOR \
            --classifiy_or_forecast "forecast" \
            --random_seed $RANDOM_SEED \
            --batch_size $BATCH_SIZE
        
        if [ $? -eq 0 ]; then
            print_success "Successfully processed $session_name for subject: $subject_id"
            print_info "Data saved to: $session_save_path"
            ((session_success_count++))
        else
            print_error "Failed to process $session_name for subject: $subject_id"
            ((session_failure_count++))
        fi
        
        # Clear GPU memory after each session to prevent accumulation
        clear_gpu_memory
    done
    
    # Report session processing results
    print_info "Subject $subject_id processing complete:"
    print_info "  Sessions processed successfully: $session_success_count"
    if [ $session_failure_count -gt 0 ]; then
        print_warning "  Sessions failed: $session_failure_count"
    fi
    
    # Return success if at least one session was processed successfully
    if [ $session_success_count -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to process subjects based on mode
process_subjects() {
    local mode=$1
    shift
    local subjects_to_process=()
    
    case $mode in
        "all")
            print_info "Processing all subjects..."
            while IFS= read -r subject_path; do
                subjects_to_process+=("$subject_path")
            done < <(get_subjects)
            ;;
            
        "random")
            local num_subjects=$1
            if [[ ! "$num_subjects" =~ ^[0-9]+$ ]] || [ "$num_subjects" -le 0 ]; then
                print_error "Invalid number of subjects: $num_subjects"
                exit 1
            fi
            
            local total_subjects=$(count_subjects)
            if [ "$num_subjects" -gt "$total_subjects" ]; then
                print_warning "Requested $num_subjects subjects, but only $total_subjects available. Processing all."
                num_subjects=$total_subjects
            fi
            
            print_info "Processing $num_subjects random subjects..."
            while IFS= read -r subject_path; do
                subjects_to_process+=("$subject_path")
            done < <(get_subjects | shuf | head -n "$num_subjects")
            ;;
            
        "specific")
            print_info "Processing specific subjects: $*"
            for subject_id in "$@"; do
                # Add sub- prefix if not present
                if [[ ! "$subject_id" =~ ^sub- ]]; then
                    subject_id="sub-$subject_id"
                fi
                
                subject_path="$BASE_PATH/$subject_id"
                if [ -d "$subject_path" ]; then
                    subjects_to_process+=("$subject_path")
                else
                    print_warning "Subject not found: $subject_id"
                fi
            done
            ;;
            
        "interactive")
            print_info "Available subjects:"
            local subjects_array=()
            local counter=1
            
            while IFS= read -r subject_path; do
                local subject_name=$(basename "$subject_path")
                subjects_array+=("$subject_path")
                echo "  $counter) $subject_name"
                ((counter++))
            done < <(get_subjects)
            
            echo ""
            echo "Enter subject numbers (space-separated, e.g., '1 3 5') or 'all' for all subjects:"
            read -r selection
            
            if [ "$selection" = "all" ]; then
                subjects_to_process=("${subjects_array[@]}")
            else
                for num in $selection; do
                    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#subjects_array[@]}" ]; then
                        subjects_to_process+=("${subjects_array[$((num-1))]}")
                    else
                        print_warning "Invalid selection: $num"
                    fi
                done
            fi
            ;;
            
        *)
            print_error "Unknown mode: $mode"
            show_usage
            exit 1
            ;;
    esac
    
    if [ ${#subjects_to_process[@]} -eq 0 ]; then
        print_error "No subjects to process!"
        exit 1
    fi
    
    print_info "Will process ${#subjects_to_process[@]} subjects"
    
    # Process each subject
    local success_count=0
    local failure_count=0
    local total_sessions_processed=0
    
    for subject_path in "${subjects_to_process[@]}"; do
        echo ""
        echo "========================================"
        local subject_id=$(basename "$subject_path" | sed 's/sub-//')
        local session_count=$(find "$subject_path" -maxdepth 1 -type d -name "ses-*" | wc -l)
        
        if process_subject "$subject_path"; then
            ((success_count++))
            ((total_sessions_processed += session_count))
        else
            ((failure_count++))
        fi
        echo "========================================"
    done
    
    echo ""
    print_info "Processing complete!"
    print_success "Successfully processed: $success_count subjects"
    print_success "Total sessions processed: $total_sessions_processed"
    if [ $failure_count -gt 0 ]; then
        print_error "Failed to process: $failure_count subjects"
    fi
}

# Main script logic
main() {
    print_info "TOTEM Zero-Shot Forecasting Data Extraction"
    print_info "============================================="
    
    # Check if arguments provided
    if [ $# -eq 0 ]; then
        print_error "No arguments provided!"
        show_usage
        exit 1
    fi
    
    # Check paths
    check_paths
    
    # Get total subjects count
    local total_subjects=$(count_subjects)
    print_info "Found $total_subjects subjects in $BASE_PATH"
    
    if [ $total_subjects -eq 0 ]; then
        print_error "No subjects found!"
        exit 1
    fi
    
    # Process based on mode
    process_subjects "$@"
    
    print_info "All done! Check results in: $SAVE_BASE_PATH"
}

# Run main function with all arguments
main "$@"
