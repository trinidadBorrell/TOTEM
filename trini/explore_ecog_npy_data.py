import numpy as np
import os
from pathlib import Path

def analyze_ecog_data_structure():
    """
    Explain the ECoG data structure based on the processing pipeline.
    """
    print("\n" + "="*80)
    print("ECoG DATA STRUCTURE EXPLANATION")
    print("="*80)
    
    print("\n📊 LABELS MEANING:")
    print("  • Original labels: Rest = 1, Move = 2")
    print("  • Processed labels: Rest = 0, Move = 1 (subtract 1 from original)")
    print("  • This is a binary classification task: Rest vs Movement")
    
    print("\n📏 DATA DIMENSIONS:")
    print("  • Data shape format: (examples/trials, time_steps, electrodes/channels)")
    print("  • Time steps: 1001 points")
    print("    - Sampling rate: 250 Hz")
    print("    - Segment duration: ~4 seconds (1001/250 = 4.004s)")
    print("    - Segments are 2s centered around each event")
    print("  • Electrodes: 72 channels (after excluding bad electrodes)")
    print("    - Patient 2 originally had 86 electrodes")
    print("    - 14 bad electrodes were excluded: [72,73,74,75,76,77,78,79,80,81,82,83,84,85]")
    
    print("\n🧠 DATA SPLITS:")
    print("  • Training: All data except last day, with bad instances removed")
    print("  • Validation: 10% of train/val data (randomly split)")
    print("  • Test: Data from the last day only, with bad instances removed")

def analyze_labels_distribution(data_dict):
    """
    Analyze the distribution of labels across different splits.
    """
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        labels_key = f"{split}_labels"
        if labels_key in data_dict:
            labels = data_dict[labels_key].flatten()
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{split.upper()} SET:")
            print(f"  Total samples: {len(labels)}")
            for label, count in zip(unique_labels, counts):
                label_name = "Rest" if label == 0 else "Move"
                percentage = (count / len(labels)) * 100
                print(f"  Label {label} ({label_name}): {count} samples ({percentage:.1f}%)")

def analyze_data_dimensions(data_dict):
    """
    Provide detailed analysis of data dimensions.
    """
    print("\n" + "="*60)
    print("DETAILED DIMENSION ANALYSIS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        data_key = f"{split}_data"
        if data_key in data_dict:
            data = data_dict[data_key]
            print(f"\n{split.upper()} DATA:")
            print(f"  Shape: {data.shape}")
            print(f"  • Dimension 0 (Examples/Trials): {data.shape[0]}")
            print(f"    - Each example is one trial/event")
            print(f"  • Dimension 1 (Time steps): {data.shape[1]}")
            print(f"    - Temporal samples at 250 Hz")
            print(f"    - Duration: {data.shape[1]/250:.3f} seconds")
            print(f"  • Dimension 2 (Electrodes): {data.shape[2]}")
            print(f"    - ECoG electrode channels")
            print(f"  • Total data points: {np.prod(data.shape):,}")
            print(f"  • Memory size: {data.nbytes / (1024**2):.1f} MB")

def sample_data_inspection(data_dict):
    """
    Show sample data from each split to understand the structure better.
    """
    print("\n" + "="*60)
    print("SAMPLE DATA INSPECTION")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        data_key = f"{split}_data"
        labels_key = f"{split}_labels"
        
        if data_key in data_dict and labels_key in data_dict:
            data = data_dict[data_key]
            labels = data_dict[labels_key]
            
            print(f"\n{split.upper()} SET SAMPLE:")
            print(f"  First trial data shape: {data[0].shape}")
            print(f"  First trial label: {labels[0]} ({'Rest' if labels[0] == 0 else 'Move'})")
            print(f"  Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
            print(f"  First electrode, first 5 timepoints: {data[0, :5, 0]}")

def read_npy_files(data_dir):
    """
    Read all .npy files in the specified directory and display information about them.
    
    Args:
        data_dir (str): Path to the directory containing .npy files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist!")
        return {}
    
    npy_files = list(data_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {data_dir}")
        return {}
    
    print(f"Found {len(npy_files)} .npy files in {data_dir}:")
    print("-" * 60)
    
    data_dict = {}
    
    for npy_file in sorted(npy_files):
        try:
            print(f"\nReading: {npy_file.name}")
            data = np.load(npy_file)
            data_dict[npy_file.stem] = data
            
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  File size: {npy_file.stat().st_size / (1024*1024):.2f} MB")
            
            # Display basic statistics for numeric data
            if np.issubdtype(data.dtype, np.number):
                print(f"  Min value: {np.min(data):.6f}")
                print(f"  Max value: {np.max(data):.6f}")
                print(f"  Mean: {np.mean(data):.6f}")
                print(f"  Std: {np.std(data):.6f}")
            
            # Show a small sample of the data
            if data.ndim == 1:
                print(f"  First 5 elements: {data[:5]}")
            elif data.ndim == 2:
                print(f"  First few elements of first row: {data[0, :min(5, data.shape[1])]}")
            else:
                print(f"  Data shape: {data.shape} (multi-dimensional)")
                
        except Exception as e:
            print(f"  Error reading {npy_file.name}: {str(e)}")
    
    return data_dict

def main():
    """
    Main function to read all .npy files in data/pt2 directory.
    """
    # Adjust the path relative to the script location
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "process_zero_shot_data" / "data" / "pt2"
    
    print("=" * 80)
    print("COMPREHENSIVE ECoG DATA ANALYSIS")
    print("=" * 80)
    
    # First explain the data structure
    analyze_ecog_data_structure()
    
    # Then load and analyze the actual data
    print("\n" + "="*80)
    print("LOADING AND ANALYZING FILES")
    print("="*80)
    
    data_dict = read_npy_files(data_dir)
    
    if data_dict:
        print(f"\n{'='*60}")
        print("BASIC SUMMARY")
        print(f"{'='*60}")
        print(f"Total files loaded: {len(data_dict)}")
        
        # Separate data and labels
        data_files = [k for k in data_dict.keys() if 'data' in k]
        label_files = [k for k in data_dict.keys() if 'labels' in k]
        
        if data_files:
            print(f"\nData files: {data_files}")
        if label_files:
            print(f"Label files: {label_files}")
            
        # Check if data and labels match
        for split in ['train', 'val', 'test']:
            data_key = f"{split}_data"
            labels_key = f"{split}_labels"
            
            if data_key in data_dict and labels_key in data_dict:
                data_samples = data_dict[data_key].shape[0]
                label_samples = data_dict[labels_key].shape[0]
                print(f"\n{split.capitalize()} set:")
                print(f"  Data samples: {data_samples}")
                print(f"  Label samples: {label_samples}")
                print(f"  Match: {'✓' if data_samples == label_samples else '✗'}")
        
        # Detailed analyses
        analyze_labels_distribution(data_dict)
        analyze_data_dimensions(data_dict)
        sample_data_inspection(data_dict)

if __name__ == "__main__":
    main()
