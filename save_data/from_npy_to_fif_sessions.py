#!/usr/bin/env python3
"""
Session-aware script to convert reconstructed .npy files to .fif format.

This script takes x_reverted.npy and y_reverted.npy files from VQVAE reconstruction,
splits them back into individual sessions based on original trial counts, and creates
separate .fif files for each session while preserving all original metadata.

Usage:
    python from_npy_to_fif_sessions.py --subject_id AA048 [--output_dir path]

Author: Generated for TOTEM project
"""

import os
import argparse
import numpy as np
import glob
import shutil
from pathlib import Path

try:
    import mne
    mne.set_log_level('WARNING')  # Reduce verbosity
except ImportError:
    print("Error: MNE-Python is required. Install it with: pip install mne")
    exit(1)


def discover_subject_sessions(subject_id, original_data_base=None):
    """
    Discover all sessions for a given subject and their trial counts.
    Handles both CONTROL and DOC subjects by checking multiple possible locations.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'AA048')
    original_data_base : str, optional
        Base directory for original data. If None, auto-detects based on subject type.
        
    Returns:
    --------
    dict : Dictionary with session info {session: {'fif_path': path, 'n_trials': count}}
    """
    # Define possible base directories for different subject types
    possible_bases = []
    
    if original_data_base:
        possible_bases.append(original_data_base)
    else:
        # Auto-detect based on common patterns
        possible_bases = [
            "/data/project/eeg_foundation/data/nice_derivatives/CONTROL_BIDS/nice_epochs_sfreq-100Hz_recombine-biosemi64",
            "/data/project/eeg_foundation/data/nice_derivatives/DOC_BIDS/nice_epochs_sfreq-100Hz_recombine-biosemi64",
            "/data/project/eeg_foundation/data/data_local_global/nice_data_raw"  # Legacy location
        ]
    
    subject_path = None
    # Try each possible base directory
    for base_dir in possible_bases:
        candidate_path = os.path.join(base_dir, f"sub-{subject_id}")
        if os.path.exists(candidate_path):
            subject_path = candidate_path
            print(f"Found subject {subject_id} in: {base_dir}")
            break
    
    if not subject_path:
        raise FileNotFoundError(f"Subject {subject_id} not found in any of: {possible_bases}")
    
    session_info = {}
    
    # Find all session directories
    for item in sorted(os.listdir(subject_path)):
        if item.startswith('ses-') and os.path.isdir(os.path.join(subject_path, item)):
            eeg_path = os.path.join(subject_path, item, 'eeg')
            if os.path.exists(eeg_path):
                # Find lg .fif files
                lg_files = glob.glob(os.path.join(eeg_path, "*lg*.fif"))
                if lg_files:
                    fif_path = lg_files[0]  # Take first if multiple
                    
                    # Load epochs to get trial count
                    epochs = mne.read_epochs(fif_path, verbose=False)
                    n_trials = len(epochs)
                    
                    session_info[item] = {
                        'fif_path': fif_path,
                        'n_trials': n_trials,
                        'epochs': epochs
                    }
                    print(f"Found {item}: {n_trials} trials")
    
    return session_info


def load_reconstructed_data(forecast_dir):
    """
    Load and concatenate x_reverted.npy and y_reverted.npy files.
    
    Parameters:
    -----------
    forecast_dir : str
        Directory containing the reconstructed .npy files
        
    Returns:
    --------
    np.ndarray : Concatenated data with shape (trials, time, sensors)
    """
    x_path = os.path.join(forecast_dir, "x_reverted.npy")
    y_path = os.path.join(forecast_dir, "y_reverted.npy")
    
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"x_reverted.npy not found in {forecast_dir}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y_reverted.npy not found in {forecast_dir}")
    
    print(f"Loading {x_path}")
    x_data = np.load(x_path)
    print(f"x_reverted shape: {x_data.shape}")
    
    print(f"Loading {y_path}")
    y_data = np.load(y_path)
    print(f"y_reverted shape: {y_data.shape}")
    
    # Concatenate along time dimension (axis=1)
    reconstructed_data = np.concatenate([x_data, y_data], axis=1)
    print(f"Concatenated data shape: {reconstructed_data.shape}")
    
    return reconstructed_data


def split_data_by_sessions(reconstructed_data, session_info):
    """
    Split reconstructed data back into individual sessions.
    
    Parameters:
    -----------
    reconstructed_data : np.ndarray
        Concatenated reconstructed data with shape (trials, time, sensors)
    session_info : dict
        Session information from discover_subject_sessions
        
    Returns:
    --------
    dict : Dictionary with session data {session: reconstructed_data_slice}
    """
    print("\nSplitting reconstructed data by sessions...")
    
    session_data = {}
    start_idx = 0
    
    # Sessions should be processed in order
    for session in sorted(session_info.keys()):
        n_trials = session_info[session]['n_trials']
        end_idx = start_idx + n_trials
        
        session_data[session] = reconstructed_data[start_idx:end_idx]
        
        print(f"{session}: trials {start_idx}:{end_idx} -> shape {session_data[session].shape}")
        start_idx = end_idx
    
    # Sanity check
    total_original_trials = sum(info['n_trials'] for info in session_info.values())
    total_reconstructed_trials = reconstructed_data.shape[0]
    
    print(f"\nSanity check:")
    print(f"Total original trials: {total_original_trials}")
    print(f"Total reconstructed trials: {total_reconstructed_trials}")
    print(f"Match: {total_original_trials == total_reconstructed_trials}")
    
    if total_original_trials != total_reconstructed_trials:
        raise ValueError(f"Trial count mismatch! Original: {total_original_trials}, Reconstructed: {total_reconstructed_trials}")
    
    return session_data


def create_session_fif_file(session, session_data_slice, original_epochs, output_dir, subject_id, session_info, scaling_factor=1e-5):
    """
    Create a .fif file for a specific session with reconstructed data.
    
    Parameters:
    -----------
    session : str
        Session name (e.g., 'ses-01')
    session_data_slice : np.ndarray
        Reconstructed data for this session (trials, time, sensors)
    original_epochs : mne.Epochs
        Original epochs object for this session
    output_dir : str
        Output directory
    subject_id : str
        Subject ID
    session_info : dict
        Session information from discover_subject_sessions
    scaling_factor : float, optional
        Factor to scale reconstructed data to match original magnitude (default: 1e-5)
        
    Returns:
    --------
    str : Path to created .fif file
    """
    print(f"\nCreating .fif file for {session}...")
    
    original_data = original_epochs.get_data()
    print(f"Original {session} shape: {original_data.shape}")
    print(f"Reconstructed {session} shape: {session_data_slice.shape}")
    
    # Validate dimensions
    n_trials_orig, n_channels_orig, n_times_orig = original_data.shape
    n_trials_recon, n_times_recon, n_channels_recon = session_data_slice.shape
    
    if n_trials_orig != n_trials_recon:
        raise ValueError(f"{session}: Trial count mismatch - original: {n_trials_orig}, reconstructed: {n_trials_recon}")
    
    if n_channels_orig != n_channels_recon:
        raise ValueError(f"{session}: Channel count mismatch - original: {n_channels_orig}, reconstructed: {n_channels_recon}")
    
    # Handle time dimension differences
    if n_times_orig != n_times_recon:
        print(f"Warning: {session} time dimension mismatch - original: {n_times_orig}, reconstructed: {n_times_recon}")
        
        if n_times_recon > n_times_orig:
            print(f"Truncating reconstructed data for {session}")
            session_data_slice = session_data_slice[:, :n_times_orig, :]
        else:
            print(f"Padding reconstructed data for {session}")
            padding = np.zeros((n_trials_recon, n_times_orig - n_times_recon, n_channels_recon))
            session_data_slice = np.concatenate([session_data_slice, padding], axis=1)
    
    # Scale reconstructed data to match original magnitude (convert to appropriate units for MNE)
    print(f"Scaling reconstructed data by {scaling_factor} for {session}")
    session_data_slice = session_data_slice * scaling_factor
    
    # Transpose to MNE format (trials, channels, time)
    reconstructed_data_mne = session_data_slice.transpose(0, 2, 1)
    
    # Create new epochs object
    reconstructed_epochs = mne.EpochsArray(
        reconstructed_data_mne,
        info=original_epochs.info.copy(),
        events=original_epochs.events.copy(),
        tmin=original_epochs.tmin,
        event_id=original_epochs.event_id.copy() if original_epochs.event_id else None,
        verbose=False
    )
    
    # Copy additional metadata
    reconstructed_epochs.metadata = original_epochs.metadata.copy() if original_epochs.metadata is not None else None
    
    # Add annotation
    original_filename = os.path.basename(original_epochs.filename) if hasattr(original_epochs, 'filename') else f"{subject_id}_{session}_lg_recon"
    reconstructed_epochs.info['description'] = f"Reconstructed from VQVAE - Session: {session}"
    
    # Create output filename for reconstructed data
    original_fif_name = os.path.basename(session_info[session]['fif_path']) if session in session_info else f"sub-{subject_id}_{session}_task-lg_acq-01_epo.fif"
    base_name = original_fif_name.replace('.fif', '')
    recon_filename = f"{base_name}_recon.fif"
    recon_output_path = os.path.join(output_dir, recon_filename)
    
    # Save reconstructed FIF file
    print(f"Saving reconstructed to: {recon_output_path}")
    reconstructed_epochs.save(recon_output_path, overwrite=True)
    
    # Also copy the original FIF file to the same directory
    orig_filename = f"{base_name}_original.fif"
    orig_output_path = os.path.join(output_dir, orig_filename)
    if session in session_info:
        print(f"Copying original to: {orig_output_path}")
        shutil.copy2(session_info[session]['fif_path'], orig_output_path)
    
    return recon_output_path, orig_output_path


def discover_subjects(forecast_base_dir):
    """Discover all available subjects."""
    subjects = []
    if not os.path.exists(forecast_base_dir):
        return subjects
    
    pattern = os.path.join(forecast_base_dir, "Tin85_Tout69_forecast_*")
    forecast_dirs = glob.glob(pattern)
    
    for forecast_dir in forecast_dirs:
        dir_name = os.path.basename(forecast_dir)
        if dir_name.startswith("Tin85_Tout69_forecast_"):
            subject_id = dir_name.replace("Tin85_Tout69_forecast_", "")
            x_path = os.path.join(forecast_dir, "x_reverted.npy")
            y_path = os.path.join(forecast_dir, "y_reverted.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                subjects.append(subject_id)
    
    return sorted(subjects)


def process_all_subject_sessions(subject_id, npy_data_base, original_data_dir, output_dir, scaling_factor=1e-5):
    """
    Process all sessions for a subject automatically.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'CH198')
    npy_data_base : str
        Base directory containing subject NPY data
    original_data_dir : str
        Base directory for original data (to load .fif metadata)
    output_dir : str
        Output directory for the reconstructed .fif files
        
    Returns:
    --------
    int : 0 if successful, 1 if failed
    """
    print(f"[INFO] Processing all sessions for subject {subject_id}...")
    
    try:
        # Discover session info for this subject
        session_info = discover_subject_sessions(subject_id, original_data_dir)
        
        if not session_info:
            raise ValueError(f"No sessions found for subject {subject_id}")
        
        print(f"Found {len(session_info)} sessions: {list(session_info.keys())}")
        
        # Find subject directory in NPY data
        subject_npy_dir = os.path.join(npy_data_base, f"sub-{subject_id}")
        if not os.path.exists(subject_npy_dir):
            raise FileNotFoundError(f"Subject NPY directory not found: {subject_npy_dir}")
        
        successful_sessions = []
        failed_sessions = []
        
        # Process each session
        for session_id in sorted(session_info.keys()):
            session_dir = os.path.join(subject_npy_dir, session_id)
            
            if not os.path.exists(session_dir):
                print(f"[WARNING] Session directory not found: {session_dir}")
                failed_sessions.append(session_id)
                continue
            
            try:
                result = process_single_session(subject_id, session_id, session_dir, original_data_dir, output_dir, scaling_factor)
                if result == 0:
                    successful_sessions.append(session_id)
                else:
                    failed_sessions.append(session_id)
            except Exception as e:
                print(f"[ERROR] Failed to process {session_id}: {e}")
                failed_sessions.append(session_id)
        
        print(f"\n[INFO] Subject {subject_id} processing complete:")
        print(f"[INFO]   Sessions converted successfully: {len(successful_sessions)}")
        if failed_sessions:
            print(f"[WARNING]   Sessions failed: {len(failed_sessions)} - {failed_sessions}")
        
        return 0 if successful_sessions else 1
        
    except Exception as e:
        print(f"[ERROR] Failed to process subject {subject_id}: {e}")
        return 1


def process_single_session(subject_id, session_id, session_dir, original_data_dir, output_dir, scaling_factor=1e-5):
    """
    Process a single session for a specific subject.
    
    Expected file structure in session_dir:
    - original.npy: original data
    - reverted.npy: reconstructed data from VQVAE
    - codes.npy: quantized codes  
    - codebook.npy: VQVAE codebook
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'CH198')
    session_id : str
        Session ID (e.g., 'ses-02')
    session_dir : str
        Directory containing the session-specific .npy files (original.npy, reverted.npy, etc.)
    original_data_dir : str
        Base directory for original data (to load .fif metadata)
    output_dir : str
        Output directory for the reconstructed .fif files
        
    Returns:
    --------
    int : 0 if successful, 1 if failed
    """
    print(f"[INFO] Converting {session_id} for subject {subject_id} to FIF format...")
    
    try:
        # Discover session info for this specific session
        session_info = discover_subject_sessions(subject_id, original_data_dir)
        
        if not session_info or session_id not in session_info:
            raise ValueError(f"Session {session_id} not found for subject {subject_id}")
        
        # Load reconstructed data from the specific session directory
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        # Expected file structure for each session:
        # - original.npy: original data
        # - reverted.npy: reconstructed data from VQVAE
        # - codes.npy: quantized codes
        # - codebook.npy: VQVAE codebook
        
        expected_files = {
            'original': os.path.join(session_dir, "original.npy"),
            'reverted': os.path.join(session_dir, "reverted.npy"),
            'codes': os.path.join(session_dir, "codes.npy"),
            'codebook': os.path.join(session_dir, "codebook.npy")
        }
        
        # Check that all expected files exist
        missing_files = []
        for file_type, file_path in expected_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}.npy")
        
        if missing_files:
            raise FileNotFoundError(f"Missing files in {session_dir}: {', '.join(missing_files)}")
        
        # Load the reconstructed data
        print(f"Loading reconstructed data from: {expected_files['reverted']}")
        reconstructed_data = np.load(expected_files['reverted'])
        print(f"Loaded reconstructed data shape: {reconstructed_data.shape}")
        
        # Optionally load original data for comparison (not used in FIF creation but useful for debugging)
        original_data = np.load(expected_files['original'])
        print(f"Original data shape: {original_data.shape}")
        
        # Create session data dict for this single session
        session_data = {session_id: reconstructed_data}
        
        # Create output directory with proper structure: output_dir/sub-{id}/ses-{num}/
        if output_dir:
            session_output_dir = os.path.join(output_dir, f"sub-{subject_id}", session_id)
        else:
            session_output_dir = f"./sub-{subject_id}/{session_id}"
        
        os.makedirs(session_output_dir, exist_ok=True)
        
        # Load original epochs for this session
        original_epochs = session_info[session_id]['epochs']
        
        # Process the session
        recon_path, orig_path = create_session_fif_file(
            session_id,
            reconstructed_data,
            original_epochs,
            session_output_dir,
            subject_id,
            session_info,
            scaling_factor
        )
        
        print(f"✓ Created reconstructed: {recon_path}")
        print(f"✓ Created original: {orig_path}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {session_id} for subject: {subject_id}")
        print(f"Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Convert reconstructed .npy files to session-specific .fif files')
    parser.add_argument('--subject_id', help='Subject ID (e.g., AA048). If not provided, processes all discovered subjects.')
    parser.add_argument('--all_subjects', action='store_true', help='Process all discovered subjects')
    parser.add_argument('--session_id', help='Session ID (e.g., ses-02). If provided, only processes this specific session.')
    parser.add_argument('--session_dir', help='Directory containing session-specific .npy files (original.npy, reverted.npy, codes.npy, codebook.npy)')
    parser.add_argument('--npy_data_base', 
                        default="/data/project/eeg_foundation/data/zero_shot_data/pydata",
                        help='Base directory for NPY data with session structure')
    parser.add_argument('--forecast_dir', help='Directory containing the reconstructed .npy files')
    parser.add_argument('--original_data_dir', 
                        help='Base directory for original data (auto-detects if not provided)')
    parser.add_argument('--output_dir', help='Output directory for the reconstructed .fif files')
    parser.add_argument('--forecast_base_dir',
                        default="/home/triniborrell/home/projects/TOTEM/forecasting/data/all_vqvae_extracted/nice",
                        help='Base directory for forecast data')
    parser.add_argument('--scaling_factor', type=float, default=1e-5,
                        help='Scaling factor to apply to reconstructed data (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Handle specific session processing
    if args.session_id and args.session_dir:
        # Process a specific session for a specific subject
        if not args.subject_id:
            print("Error: --subject_id must be provided when using --session_id and --session_dir")
            return 1
        
        return process_single_session(args.subject_id, args.session_id, args.session_dir, 
                                    args.original_data_dir, args.output_dir, args.scaling_factor)
    
    # Handle processing all sessions for a subject
    if args.subject_id and not args.session_id:
        return process_all_subject_sessions(args.subject_id, args.npy_data_base,
                                           args.original_data_dir, args.output_dir, args.scaling_factor)
    
    # Determine which subjects to process
    if args.all_subjects or not args.subject_id:
        subjects = discover_subjects(args.forecast_base_dir)
        if not subjects:
            print(f"No subjects found in {args.forecast_base_dir}")
            return 1
        print(f"Discovered subjects: {subjects}")
        if not args.all_subjects and not args.subject_id:
            print("Use --subject_id <ID> to process a specific subject or --all_subjects to process all")
            return 1
    else:
        subjects = [args.subject_id]
    
    successful = []
    failed = []
    
    print(f"Processing {len(subjects)} subject(s) with session splitting...")
    
    for subject_id in subjects:
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT: {subject_id}")
        print(f"{'='*80}")
        
        try:
            # Determine forecast directory
            if args.forecast_dir:
                forecast_dir = args.forecast_dir
            else:
                forecast_dir = os.path.join(args.forecast_base_dir, f"Tin85_Tout69_forecast_{subject_id}")
            
            # Discover sessions for this subject
            print(f"Discovering sessions for subject {subject_id}...")
            session_info = discover_subject_sessions(subject_id, args.original_data_dir)
            
            if not session_info:
                raise ValueError(f"No sessions found for subject {subject_id}")
            
            print(f"Found {len(session_info)} sessions: {list(session_info.keys())}")
            
            # Load reconstructed data
            reconstructed_data = load_reconstructed_data(forecast_dir)
            
            # Split data by sessions
            session_data = split_data_by_sessions(reconstructed_data, session_info)
            
            # Determine output directory
            if args.output_dir:
                output_dir = os.path.join(args.output_dir, f"sub-{subject_id}")
            else:
                output_dir = os.path.join(forecast_dir, f"sub-{subject_id}_sessions")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Create .fif files for each session
            created_files = []
            for session in sorted(session_info.keys()):
                try:
                    # Create session output directory
                    session_output_dir = os.path.join(output_dir, session)
                    os.makedirs(session_output_dir, exist_ok=True)
                    
                    recon_path, orig_path = create_session_fif_file(
                        session,
                        session_data[session],
                        session_info[session]['epochs'],
                        session_output_dir,
                        subject_id,
                        session_info,
                        args.scaling_factor
                    )
                    created_files.extend([recon_path, orig_path])
                    print(f"✓ Created reconstructed: {recon_path}")
                    print(f"✓ Created original: {orig_path}")
                except Exception as e:
                    print(f"✗ Failed to create {session}: {e}")
                    raise
            
            # Summary for this subject
            print(f"\n{'='*60}")
            print(f"SUBJECT {subject_id} SUMMARY")
            print(f"{'='*60}")
            print(f"Sessions processed: {len(created_files)}")
            for file_path in created_files:
                session = os.path.basename(os.path.dirname(file_path))
                filename = os.path.basename(file_path)
                print(f"  {session}: {filename}")
            print(f"Output directory: {output_dir}")
            
            successful.append(subject_id)
            
        except Exception as e:
            print(f"✗ Error processing subject {subject_id}: {e}")
            failed.append((subject_id, str(e)))
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(successful)} subjects")
    for s in successful:
        print(f"  ✓ {s}")
    
    if failed:
        print(f"\nFailed to process: {len(failed)} subjects")
        for s, error in failed:
            print(f"  ✗ {s}: {error}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main()) 