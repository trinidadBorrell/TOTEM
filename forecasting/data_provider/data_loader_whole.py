import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Neuro_Whole(Dataset):
    """
    Dataset loader for processing whole/continuous EEG data without windowing.
    Loads the complete data.npy file and processes it as continuous sequences.
    """
    def __init__(self, root_path, flag='test', size=None,
                 features='M', data_path='neuro_zero_shot',
                 target='OT', scale=True, timeenc=0, freq='h'):
        
        # For whole data processing, we don't need size parameters
        # but keep them for compatibility
        if size is not None:
            self.seq_len = size[0]
            self.label_len = size[1] 
            self.pred_len = size[2]
        else:
            self.seq_len = None
            self.label_len = None
            self.pred_len = None
        
        # For whole data processing, we typically work with test data
        assert flag == 'test'
        self.set_type = 2  # Always test

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Load raw data - try multiple naming patterns
        data = None
        data_file_patterns = [
            'data.npy',  # New naming convention from process_eeg_data_zero_shot.py
            'test_data.npy',  # Old naming convention
            f'{self.data_path}_data.npy',  # With prefix, new convention
            f'{self.data_path}_test_data.npy',  # With prefix, old convention
        ]
        
        for pattern in data_file_patterns:
            try:
                data = np.load(os.path.join(self.root_path, pattern))
                print(f"Loaded whole data from {pattern}: {data.shape}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            if os.path.exists(self.root_path):
                files = os.listdir(self.root_path)
                print(f"Available files in {self.root_path}: {files}")
                print(f"Tried patterns: {data_file_patterns}")
            raise FileNotFoundError(f"Could not find data file in {self.root_path}")

        # Data shape should be [trials/epochs, time_points, channels]
        print(f"Raw data shape: {data.shape}")
        print(f"Data will be processed as continuous sequences without windowing")
        
        # Store the raw data - each trial/epoch will be processed separately
        self.data = data
        
        # For scaling, reshape to [all_timepoints, channels] if needed
        if self.scale:
            # Reshape for fitting scaler: [trials*time, channels]
            data_reshaped = data.reshape(-1, data.shape[-1])
            self.scaler.fit(data_reshaped)
            print(f"Fitted scaler on data with shape: {data_reshaped.shape}")
        
        print(f"Loaded {len(self.data)} trials/epochs for whole data processing")

    def __getitem__(self, index):
        # Return one complete trial/epoch at a time
        # Shape: [time_points, channels]
        trial_data = self.data[index]
        
        if self.scale:
            # Scale this trial
            trial_data_scaled = self.scaler.transform(trial_data)
            return trial_data_scaled, trial_data_scaled, 0, 0  # Return same data for both x and y
        else:
            return trial_data, trial_data, 0, 0

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Neuro_Whole_Single(Dataset):
    """
    Dataset loader for processing ALL data as a single continuous sequence.
    Concatenates all trials/epochs into one long sequence.
    """
    def __init__(self, root_path, flag='test', size=None,
                 features='M', data_path='neuro_zero_shot',
                 target='OT', scale=True, timeenc=0, freq='h'):
        
        # For single sequence processing, size is not relevant
        self.seq_len = None
        self.label_len = None
        self.pred_len = None
        
        assert flag == 'test'
        self.set_type = 2

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Load raw data - try multiple naming patterns
        data = None
        data_file_patterns = [
            'data.npy',
            'test_data.npy', 
            f'{self.data_path}_data.npy',
            f'{self.data_path}_test_data.npy',
        ]
        
        for pattern in data_file_patterns:
            try:
                data = np.load(os.path.join(self.root_path, pattern))
                print(f"Loaded data from {pattern}: {data.shape}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            if os.path.exists(self.root_path):
                files = os.listdir(self.root_path)
                print(f"Available files in {self.root_path}: {files}")
                print(f"Tried patterns: {data_file_patterns}")
            raise FileNotFoundError(f"Could not find data file in {self.root_path}")

        # Concatenate all trials into one long sequence
        # From [trials, time, channels] to [total_time, channels]
        print(f"Original data shape: {data.shape}")
        
        # Reshape to concatenate all trials
        total_time = data.shape[0] * data.shape[1]
        channels = data.shape[2]
        concatenated_data = data.reshape(total_time, channels)
        
        print(f"Concatenated data shape: {concatenated_data.shape}")
        print(f"Total time points: {total_time}, Channels: {channels}")
        
        # Store the concatenated data
        self.data = concatenated_data
        
        # Fit scaler on the concatenated data
        if self.scale:
            self.scaler.fit(self.data)
            print(f"Fitted scaler on concatenated data")

    def __getitem__(self, index):
        # For single sequence, we only have one item (index should be 0)
        if index != 0:
            raise IndexError("Single sequence dataset only has one item")
            
        if self.scale:
            data_scaled = self.scaler.transform(self.data)
            return data_scaled, data_scaled, 0, 0
        else:
            return self.data, self.data, 0, 0

    def __len__(self):
        return 1  # Only one long sequence

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)