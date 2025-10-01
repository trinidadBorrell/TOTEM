BASE_PATH="/home/triniborrell/home/data/nice_data_raw"
SAVE_PATH="/home/triniborrell/home/data/nice_data_processed"
N_SUBJECTS=3
RANDOM_SEED=42

# Run the processing script
python process_zero_shot_data/process_eeg_data_zero_shot.py \
    --base_path "$BASE_PATH" \
    --save_path "$SAVE_PATH" \
    --n_subjects $N_SUBJECTS \


echo "EEG data processing completed!"
echo "Processed data saved to: $SAVE_PATH"