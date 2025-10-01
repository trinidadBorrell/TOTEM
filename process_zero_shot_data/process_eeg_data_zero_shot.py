#!/usr/bin/env python
# process_eeg_data_zero_shot.py
"""
Process local-global EEG data for TOTEM zero-shot learning.
Each session is saved in a separate folder structure.

Output structure:
save_path/
├── sub-{subject_id}/
│   ├── ses-01/
│   │   ├── data.npy
│   │   └── trial_labels.npy
│   ├── ses-02/
│   │   ├── data.npy
│   │   └── trial_labels.npy
│   └── ...

Usage examples
--------------
# Process first 3 subjects
python process_eeg_data_zero_shot.py \
    --base_path /path/to/eeg/data \
    --save_path /path/to/save/processed/data \
    --n_subjects 3

# Process 5 random subjects
python process_eeg_data_zero_shot.py \
    --base_path /path/to/eeg/data \
    --save_path /path/to/save/processed/data \
    --n_subjects 5 --random

# Process specific subjects
python process_eeg_data_zero_shot.py \
    --base_path /path/to/eeg/data \
    --save_path /path/to/save/processed/data \
    --subjects PD155 LP275 AA069

# Process all subjects
python process_eeg_data_zero_shot.py \
    --base_path /path/to/eeg/data \
    --save_path /path/to/save/processed/data \
    --n_subjects 999999
"""
import argparse
import os
import random
import re
import warnings
from pathlib import Path

import mne
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# 1.  EVENT CODE → DESCRIPTION MAPPING
# ────────────────────────────────────────────────────────────────────────────────
event_dict = {
    10: 'CTRL1',   # Control 1
    20: 'CTRL2',   # Control 2  
    30: 'LSGS',    # Local Standard Global Standard (highest count)
    40: 'LDGD',    # Local Deviant Global Deviant
    50: 'LSGD',    # Local Standard Global Deviant
    60: 'LDGS'     # Local Deviant Global Standard
}
# ────────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ────────────────────────────────────────────────────────────────────────────────
def load_eeg_data(subject_path: Path, task: str):
    """
    Load EEG epochs for one subject & task, keeping sessions separate.
    
    Automatically handles different channel configurations:
    - 64 or 256 channels: kept as-is
    - < 64 channels: zero-padded to 64
    - 64 < channels < 256: truncated to 64 (if ≤128) or padded to 256 (if >128)
    - > 256 channels: truncated to 256

    Returns
    -------
    sessions_data : dict
                   {session_num: {'data': (n_epochs, n_channels, n_times) float32,
                                 'labels': (n_epochs, 3) object}}
                   where labels columns are:
                   ⎡event_code  int16,
                    session     int16,
                    description str⎤
    """
    fif_files = list(
        Path(subject_path).glob(f"**/sub-*_ses-*_task-{task}_acq-*_epo.fif")
    )
    if not fif_files:
        print(f"No files found for task {task} in {subject_path}")
        return None

    ses_re = re.compile(r"_ses-(\d+)_")
    sessions_data = {}

    for fif in fif_files:
        print(f"Loading {fif}")
        session_num = int(ses_re.search(fif.name).group(1))

        # -- read epochs with error handling --------------------------------------------------------
        try:
            # Check file size first
            file_size = os.path.getsize(fif)
            if file_size == 0:
                print(f"  Warning: File {fif} is empty, skipping...")
                continue
            
            epochs = mne.read_epochs(fif, verbose=False)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = epochs.get_data()  # (n_ep, n_ch, n_t)
                
        except Exception as e:
            print(f"  Error loading {fif}: {e}")
            print(f"  File may be corrupted. Skipping session {session_num}...")
            continue

        # Handle different channel configurations (64 or 256 channels)
        n_channels = data.shape[1]
        expected_channels = None
        
        if n_channels in [64, 256]:
            # Standard configurations - keep as is
            expected_channels = n_channels
            print(f"  Found {n_channels} channels (standard configuration)")
        elif n_channels < 64:
            # Pad to 64 channels
            expected_channels = 64
            print(f"  Warning: {n_channels} < 64 channels, zero-padding to 64")
            padded = np.zeros((data.shape[0], 64, data.shape[2]), data.dtype)
            padded[:, :n_channels, :] = data
            data = padded
        elif 64 < n_channels < 256:
            # Truncate to 64 or pad to 256 based on proximity
            if n_channels <= 128:
                expected_channels = 64
                print(f"  Warning: {n_channels} channels found, truncating to first 64")
                data = data[:, :64, :]
            else:
                expected_channels = 256
                print(f"  Warning: {n_channels} channels found, zero-padding to 256")
                padded = np.zeros((data.shape[0], 256, data.shape[2]), data.dtype)
                padded[:, :n_channels, :] = data
                data = padded
        else:
            # More than 256 channels - truncate to 256
            expected_channels = 256
            print(f"  Warning: {n_channels} channels found, truncating to first 256")
            data = data[:, :256, :]

        # -- labels ------------------------------------------------------------
        labels_numeric = epochs.events[:, 2].astype(np.int16)  # 10,20,30...
        labels_desc = np.vectorize(event_dict.get)(labels_numeric.astype(str))
        session_col = np.full_like(labels_numeric, session_num, dtype=np.int16)

        labels = np.column_stack((labels_numeric, session_col, labels_desc))
        
        # Store session data separately
        sessions_data[session_num] = {
            'data': data,
            'labels': labels
        }

        uniq = np.unique(labels_numeric)
        print(f"  Session {session_num}: {len(labels)} epochs – codes present: {uniq}")

    if not sessions_data:
        return None

    return sessions_data


# ────────────────────────────────────────────────────────────────────────────────
# 3.  SUBJECT-LEVEL WRAPPER
# ────────────────────────────────────────────────────────────────────────────────
def process_subject(subject_path: Path, subject_id: str):
    sessions_data = load_eeg_data(subject_path, "lg")
    if sessions_data is None:
        print(f"No local-global data for {subject_id}")
        return {}

    subject_data = {}
    for session_num, session_data in sessions_data.items():
        # (n_ep, n_t, n_ch) for downstream
        data_transposed = np.transpose(session_data["data"], (0, 2, 1))
        subject_data[session_num] = {
            "data": data_transposed,
            "labels": session_data["labels"]
        }
        print(
            f"Subject {subject_id} – Session {session_num}: {data_transposed.shape}, "
            f"labels {session_data['labels'].shape}"
        )
    return subject_data


# ────────────────────────────────────────────────────────────────────────────────
# 4.  SAVING
# ────────────────────────────────────────────────────────────────────────────────
def save_processed_data(data_dict: dict, save_path: str, subject_id: str):
    """
    Save processed data with sessions in separate folders.
    
    Structure: save_path/sub-{subject_id}/ses-{session_num}/data.npy
                                                           /trial_labels.npy
    """
    subject_dir = Path(save_path) / f"sub-{subject_id}"
    
    for session_num, session_data in data_dict.items():
        session_dir = subject_dir / f"ses-{session_num:02d}"
        session_dir.mkdir(parents=True, exist_ok=True)

        np.save(session_dir / "data.npy", session_data["data"])
        # Mixed-type array → dtype=object so NumPy keeps the strings
        np.save(session_dir / "trial_labels.npy", session_data["labels"], allow_pickle=True)

        print(
            f"Saved to {session_dir} – data {session_data['data'].shape}, "
            f"labels {session_data['labels'].shape}"
        )


# ────────────────────────────────────────────────────────────────────────────────
# 5.  MAIN SCRIPT
# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Process EEG local-global task data")
    ap.add_argument("--base_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--n_subjects", type=int, default=3, 
                    help="Number of subjects to process")
    ap.add_argument("--subjects", nargs="+", 
                    help="Specific subject IDs to process")
    ap.add_argument("--random", action="store_true", 
                    help="Randomly select n_subjects instead of taking first n")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible random selection")
    args = ap.parse_args()

    base = Path(args.base_path)
    subjects = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    print(f"Found {len(subjects)} subjects.")

    if args.subjects:
        # Process specific subjects
        wanted = {("sub-" + s if not s.startswith("sub-") else s) for s in args.subjects}
        subjects = [s for s in subjects if s in wanted]
        print(f"Processing specific subjects: {subjects}")
    else:
        # Process n_subjects (either first n or random n)
        if args.random:
            # Set random seed for reproducibility
            random.seed(args.seed)
            if len(subjects) <= args.n_subjects:
                print(f"Requested {args.n_subjects} random subjects, but only {len(subjects)} available. Processing all.")
            else:
                subjects = random.sample(subjects, args.n_subjects)
                print(f"Randomly selected {args.n_subjects} subjects (seed={args.seed}): {subjects}")
        else:
            subjects = subjects[: args.n_subjects]
            print(f"Processing first {args.n_subjects} subjects: {subjects}")

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        print("\n" + "=" * 60 + f"\nProcessing {subj}\n" + "=" * 60)
        data_dict = process_subject(base / subj, subj.replace("sub-", ""))
        if not data_dict:
            print(f"No data found for {subj}, skipping...")
            continue
        save_processed_data(data_dict, args.save_path, subj.replace("sub-", ""))

    print(f"\nDone – processed data saved under {args.save_path}")


if __name__ == "__main__":
    main()
