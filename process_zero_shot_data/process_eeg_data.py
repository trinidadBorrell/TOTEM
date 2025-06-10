import argparse
import numpy as np
import os
import mne
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
import warnings


def load_eeg_data(subject_path, task):
    """
    Load EEG data for a specific subject and task.
    
    Args:
        subject_path: Path to subject directory
        task: Either 'lg' (local-global) or 'rs' (resting state)
    
    Returns:
        epochs_data: numpy array of shape (n_epochs, n_channels, n_times)
        labels: numpy array of shape (n_epochs, 1) with task labels
    """
    # Find all .fif files for the given task
    pattern = f"**/sub-*_ses-*_task-{task}_acq-*_epo.fif"
    fif_files = list(Path(subject_path).glob(pattern))
    
    if not fif_files:
        print(f"No files found for task {task} in {subject_path}")
        return None, None
    
    all_epochs_data = []
    all_labels = []
    
    for fif_file in fif_files:
        print(f"Loading {fif_file}")
        
        # Load epochs from .fif file
        epochs = mne.read_epochs(fif_file, verbose=False)
        
        # Get the data: (n_epochs, n_channels, n_times)
        # Suppress FutureWarning about copy parameter
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = epochs.get_data()
        
        # Ensure we have exactly 64 channels
        if data.shape[1] != 64:
            if data.shape[1] > 64:
                print(f"  Warning: {data.shape[1]} channels found, taking first 64")
                data = data[:, :64, :]
            else:
                print(f"  Warning: Only {data.shape[1]} channels found, zero-padding to 64")
                # Zero-pad to 64 channels
                padded_data = np.zeros((data.shape[0], 64, data.shape[2]))
                padded_data[:, :data.shape[1], :] = data
                data = padded_data
        
        # Create labels: 1 for local-global, 0 for resting state
        if task == 'lg':
            labels = np.ones((data.shape[0], 1))
        else:  # task == 'rs'
            labels = np.zeros((data.shape[0], 1))
        
        all_epochs_data.append(data)
        all_labels.append(labels)
        
        print(f"  Loaded {data.shape[0]} epochs, {data.shape[1]} channels, {data.shape[2]} timepoints")
    
    # Concatenate all data
    if all_epochs_data:
        epochs_data = np.concatenate(all_epochs_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return epochs_data, labels
    else:
        return None, None


def process_subject(subject_path, subject_id):
    """
    Process local-global task data for a single subject.
    
    Args:
        subject_path: Path to subject directory
        subject_id: Subject identifier
    
    Returns:
        Dictionary with processed data for local-global task only
    """
    subject_data = {}
    
    # Process only local-global task
    lg_data, lg_labels = load_eeg_data(subject_path, 'lg')
    if lg_data is not None:
        subject_data['lg'] = {'data': lg_data, 'labels': lg_labels}
        print(f"Subject {subject_id} - Local-Global: {lg_data.shape}")
    else:
        print(f"No local-global data found for subject {subject_id}")
    
    return subject_data


def create_train_test_val_splits(data_dict, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/test/val splits maintaining 70% train, 20% test, 10% val.
    Only processes local-global task data.
    
    Args:
        data_dict: Dictionary containing data and labels for local-global task
        test_size: Proportion for test set (0.2 = 20%)
        val_size: Proportion for validation set (0.1 = 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train/test/val splits
    """
    # Get local-global task data only
    if 'lg' not in data_dict:
        print("No local-global data found!")
        return None
    
    lg_data = data_dict['lg']['data']
    lg_labels = data_dict['lg']['labels']
    
    # Convert to (n_epochs, n_times, n_channels) format as required
    data_transposed = np.transpose(lg_data, (0, 2, 1))
    
    print(f"Local-global data shape: {lg_data.shape} -> Final shape: {data_transposed.shape}")
    print(f"Labels shape: {lg_labels.shape}")
    print(f"All labels are 1 (local-global task): {np.unique(lg_labels)}")
    
    # Since all labels are the same (1 for local-global), we can't use stratify
    # First split: separate out test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        data_transposed, lg_labels, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Second split: from remaining 80%, take 12.5% for val (which is 10% of total)
    # and 87.5% for train (which is 70% of total)
    val_size_adjusted = val_size / (1 - test_size)  # 0.1 / 0.8 = 0.125
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    return {
        'train': {'data': X_train, 'labels': y_train},
        'test': {'data': X_test, 'labels': y_test},
        'val': {'data': X_val, 'labels': y_val}
    }


def save_processed_data(splits_dict, save_path, subject_id):
    """
    Save processed data to .npy files.
    
    Args:
        splits_dict: Dictionary with train/test/val splits
        save_path: Directory to save files
        subject_id: Subject identifier
    """
    subject_save_path = os.path.join(save_path, f'sub-{subject_id}')
    
    if not os.path.exists(subject_save_path):
        os.makedirs(subject_save_path)
    
    for split_name, split_data in splits_dict.items():
        data = split_data['data']
        labels = split_data['labels']
        
        # Save data and labels
        np.save(os.path.join(subject_save_path, f'{split_name}_data.npy'), data)
        np.save(os.path.join(subject_save_path, f'{split_name}_labels.npy'), labels)
        
        print(f"Saved {split_name} data: {data.shape}, labels: {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description='Process EEG local-global task data for TOTEM')
    parser.add_argument('--base_path', type=str, required=True, 
                       help='Base path to EEG data directory')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Directory to save processed files')
    parser.add_argument('--n_subjects', type=int, default=3,
                       help='Number of subjects to process (default: 3)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='Specific subject IDs to process (e.g., PD155 LP275 AA069)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for train/test/val splits')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Get list of available subjects
    base_path = Path(args.base_path)
    all_subjects = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    all_subjects.sort()
    
    print(f"Found {len(all_subjects)} subjects in total")
    
    # Select subjects to process
    if args.subjects:
        # Use specified subjects
        subjects_to_process = [f'sub-{s}' if not s.startswith('sub-') else s for s in args.subjects]
        subjects_to_process = [s for s in subjects_to_process if s in all_subjects]
    else:
        # Use first n_subjects
        subjects_to_process = all_subjects[:args.n_subjects]
    
    print(f"Processing {len(subjects_to_process)} subjects: {subjects_to_process}")
    
    # Process each subject
    for subject in subjects_to_process:
        print(f"\n{'='*50}")
        print(f"Processing {subject}")
        print(f"{'='*50}")
        
        subject_path = base_path / subject
        subject_id = subject.replace('sub-', '')
        
        # Process subject data
        subject_data = process_subject(subject_path, subject_id)
        
        if not subject_data:
            print(f"No data found for {subject}, skipping...")
            continue
        
        # Create train/test/val splits
        splits = create_train_test_val_splits(
            subject_data, 
            test_size=0.2, 
            val_size=0.1, 
            random_state=args.random_seed
        )
        
        if splits is None:
            print(f"Failed to create splits for {subject}, skipping...")
            continue
        
        # Save processed data
        save_processed_data(splits, args.save_path, subject_id)
        
        print(f"Successfully processed {subject}")
        print(f"  Train: {splits['train']['data'].shape}")
        print(f"  Test: {splits['test']['data'].shape}")
        print(f"  Val: {splits['val']['data'].shape}")
    
    print(f"\nProcessing complete! Data saved to {args.save_path}")


if __name__ == '__main__':
    main()
