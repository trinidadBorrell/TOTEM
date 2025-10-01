# TOTEM Zero-Shot EEG Reconstruction Pipeline

This README describes the complete 3-step pipeline for processing, reconstructing, and converting EEG data using VQVAE zero-shot learning in the TOTEM project.

## 📋 Overview

The pipeline consists of three sequential steps that transform raw EEG data through a VQVAE-based reconstruction process:

1. **Step 1**: EEG Data Preprocessing (`step1_process_nice_data.sh`)
2. **Step 2**: VQVAE Encoding & Reconstruction (`step2_zero_shot.sh`) 
3. **Step 3**: NPY to FIF Conversion (`step3_to_fif.sh`)

## 🔄 Pipeline Flow

```
Raw EEG (.fif) → Processed Data → VQVAE Codes → Reconstructed Data → Scaled FIF Files
     Step 1           Step 2           Step 3
```

---

## 📂 Data Flow & Locations

### Input Data
- **Original FIF Files**: 
  - FileStructure: `sub-{ID}/ses-{XX}/eeg/sub-{ID}_ses-{XX}_task-lg_acq-01_epo.fif`
  - Contains: Raw EEG epochs for local-global paradigm tasks

### Intermediate Data
- **Step 1 Output**: `/data/project/eeg_foundation/data/processed_nice_data/`
- **Step 2 Output**: `/data/project/eeg_foundation/data/zero_shot_data/pydata/`

### Final Output
- **Step 3 Output**: `/data/project/eeg_foundation/data/zero_shot_data/fifdata/`
  ```
  sub-{ID}/
  ├── original/          # Original FIF files (copied)
  │   └── sub-{ID}_ses-{XX}_task-lg_acq-01_epo.fif
  └── reconstruction/    # Reconstructed FIF files
      └── sub-{ID}_ses-{XX}_task-lg_acq-01_epo_reconstructed.fif
  ```

---

## 🔧 Step 1: EEG Data Preprocessing

### Purpose
Preprocesses raw EEG data for zero-shot learning by filtering, epoching, and preparing data in the format required for VQVAE processing.

### Script
`scipts/step1_process_nice_data.sh` → `process_zero_shot_data/process_eeg_data_zero_shot.py`

### Input
- Raw EEG epochs (.fif files) from NICE dataset
- Local-global paradigm task data (`task-lg`)

### Output
- Processed EEG data ready for VQVAE encoding
- Standardized format with consistent preprocessing

### Usage
```bash
# Process all subjects
./scipts/step1_process_nice_data.sh all

# Process 5 random subjects
./scipts/step1_process_nice_data.sh random 5

# Process specific subjects
./scipts/step1_process_nice_data.sh specific AA048 LP275 PD155

# Interactive mode
./scipts/step1_process_nice_data.sh interactive
```

### What it does
- Loads EEG epochs from FIF files
- Applies preprocessing (filtering, artifact removal)
- Converts to standardized format for VQVAE processing
- Handles event codes and metadata preservation

---

## 🧠 Step 2: VQVAE Encoding & Reconstruction

### Purpose
Uses a pretrained VQVAE model to encode EEG data into discrete codes and then reconstruct the signal, enabling zero-shot learning capabilities.

### Script
`scipts/step2_zero_shot.sh` → `forecasting/extract_zero_shot_data_single_df.py`

### Model
- **VQVAE Model**: `forecasting/pretrained/forecasting/checkpoints/final_model.pth`
- **Compression Factor**: 4x
- **RevIN Normalization**: Applied for stable reconstruction

### Input
- Processed EEG data from Step 1

### Output (per subject)
- `codebook.npy`: VQVAE codebook vectors
- `codes.npy`: Discrete code indices  
- `original.npy`: Original data after preprocessing
- `reverted.npy`: **Reconstructed data** (main output)

### Usage
```bash
# Process all subjects
./scipts/step2_zero_shot.sh all

# Process 5 random subjects
./scipts/step2_zero_shot.sh random 5

# Process specific subjects  
./scipts/step2_zero_shot.sh specific sub-AA048 sub-LP275

# Interactive mode
./scipts/step2_zero_shot.sh interactive
```

### What it does
- Loads processed EEG data
- Applies RevIN normalization
- Encodes data using pretrained VQVAE model
- Generates discrete codes and reconstructs signal
- Saves all intermediate and final representations

---

## 🔄 Step 3: NPY to FIF Conversion with Scaling

### Purpose
Converts reconstructed NPY data back to FIF format with proper scaling, session splitting, and metadata preservation for downstream analysis.

### Script
`scipts/step3_to_fif.sh` → `trini/scripts/from_npy_to_fif.py`

### Key Features
- **Automatic Scaling**: Applies 1e-5 scaling factor to match original EEG magnitude
- **Session Splitting**: Reconstructs individual session files from concatenated data
- **Metadata Preservation**: Maintains all original FIF metadata (events, channel info, etc.)
- **Dual Output**: Saves both original and reconstructed FIF files

### Input
- `reverted.npy` files from Step 2
- Original FIF files (for metadata reference)

### Output Structure
```
/data/project/eeg_foundation/data/zero_shot_data/fifdata/
└── sub-{ID}/
    ├── original/           # Original FIF files
    │   └── sub-{ID}_ses-{XX}_task-lg_acq-01_epo.fif
    └── reconstruction/     # Reconstructed FIF files  
        └── sub-{ID}_ses-{XX}_task-lg_acq-01_epo_reconstructed.fif
```

### Usage
```bash
# Process all subjects
./scipts/step3_to_fif.sh all

# Process 5 random subjects
./scipts/step3_to_fif.sh random 5

# Process specific subjects
./scipts/step3_to_fif.sh specific AA048 LP275

# Interactive mode
./scipts/step3_to_fif.sh interactive
```
---

## 🚀 Complete Pipeline Execution

### Run All Steps Sequentially
```bash
# Step 1: Preprocess EEG data
./scipts/step1_process_nice_data.sh all

# Step 2: VQVAE encoding and reconstruction  
./scipts/step2_zero_shot.sh all

# Step 3: Convert to FIF with scaling
./scipts/step3_to_fif.sh all
```

### Run for Specific Subjects
```bash
# Process subject AA048 through all steps
./scipts/step1_process_nice_data.sh specific AA048
./scipts/step2_zero_shot.sh specific sub-AA048  
./scipts/step3_to_fif.sh specific AA048
```

### Run Random Sample
```bash
# Process 3 random subjects through all steps
./scipts/step1_process_nice_data.sh random 3
./scipts/step2_zero_shot.sh random 3
./scipts/step3_to_fif.sh random 3
```

---

## 👥 Authors & Contact

**Trinidad Borrell** - triniborrell@gmail.com

For questions or issues, please refer to the project documentation or contact the development team.

---

## 📚 Citation

This is an adapted version of TOTEM for EEG reconstruction. Please cite the original work:

```bibtex
@article{
talukder2024totem,
title={{TOTEM}: {TO}kenized Time Series {EM}beddings for General Time Series Analysis},
author={Sabera J Talukder and Yisong Yue and Georgia Gkioxari},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=QlTLkH6xRC},
note={}
}
```

**Note**: This implementation is preliminary and may be refined in the future for forecasting applications.