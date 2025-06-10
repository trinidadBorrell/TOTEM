#!/bin/bash

# Process EEG local-global task data for TOTEM
# This script processes only local-global task EEG data from .fif files and creates train/test/val splits

# Basic parameters
BASE_PATH="C:/Users/Usuario/Documents/Trini/BackUp Asus/Trini_PhD/Trini_PhD/TOTEM_repo/nice_derivatives/nice_epochs_sfreq-100Hz_recombine-biosemi64"
SAVE_PATH="C:/Users/Usuario/Documents/Trini/BackUp Asus/Trini_PhD/Trini_PhD/TOTEM_repo/data"
N_SUBJECTS=3
RANDOM_SEED=42

# Run the processing script
python process_zero_shot_data/process_eeg_data.py \
    --base_path "$BASE_PATH" \
    --save_path "$SAVE_PATH" \
    --n_subjects $N_SUBJECTS \
    --random_seed $RANDOM_SEED

echo "EEG data processing completed!"
echo "Processed data saved to: $SAVE_PATH"

# Alternative: Process specific subjects
# Uncomment the lines below and comment the above if you want to process specific subjects
# python process_zero_shot_data/process_eeg_data.py \
#     --base_path "$BASE_PATH" \
#     --save_path "$SAVE_PATH" \
#     --subjects PD155 LP275 AA069 \
#     --random_seed $RANDOM_SEED 