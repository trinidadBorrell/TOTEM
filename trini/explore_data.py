import mne
import argparse

def read_fif_file(file_path):
    print(f"Reading FIF file: {file_path}")

    # Load epochs instead of raw data
    epochs = mne.read_epochs(file_path, preload=True)

    # Print basic info
    print("\n--- Epochs Info ---")
    print(epochs)

    print(f"\nNumber of epochs: {len(epochs.events)}")
    print(f"Epoch length (s): {epochs.times[-1]:.2f}")
    print(f"Number of channels: {len(epochs.ch_names)}")

    print("\n--- Channel Names ---")
    print(epochs.ch_names)

    return epochs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and display basic info from a .fif epochs file.")
    parser.add_argument("fif_file", type=str, help="Path to the .fif epochs file")
    args = parser.parse_args()

    epochs_data = read_fif_file(args.fif_file)

