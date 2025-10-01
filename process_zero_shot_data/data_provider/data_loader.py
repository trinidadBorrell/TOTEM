import pdb
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from process_zero_shot_data.utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Neuro(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

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
        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))

        train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        val_data_reshaped = val_data.reshape(-1, val_data.shape[-1])
        test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])

        if self.scale:
            self.scaler.fit(train_data_reshaped)
            train_data_scaled = self.scaler.transform(train_data_reshaped)
            val_data_scaled = self.scaler.transform(val_data_reshaped)
            test_data_scaled = self.scaler.transform(test_data_reshaped)

        train_scaled_orig_shape = train_data_scaled.reshape(train_data.shape)
        val_scaled_orig_shape = val_data_scaled.reshape(val_data.shape)
        test_scaled_orig_shape = test_data_scaled.reshape(test_data.shape)

        if self.set_type == 0:  # TRAIN
            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
            self.data_x = train_x
            self.data_y = train_y

        elif self.set_type == 1:  # VAL
            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
            self.data_x = val_x
            self.data_y = val_y

        elif self.set_type == 2:  # TEST
            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
            self.data_x = test_x
            self.data_y = test_y

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        pdb.set_trace()
        return self.scaler.inverse_transform(data)


class Dataset_Neuro_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='nice',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

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
        
        # Look for train/val/test files with various naming patterns
        possible_patterns = [
            # Standard pattern like Dataset_Neuro
            ('train_data.npy', 'val_data.npy', 'test_data.npy'),
            # Pattern with data_path prefix
            (f'{self.data_path}_train.npy', f'{self.data_path}_val.npy', f'{self.data_path}_test.npy'),
            (f'train_{self.data_path}.npy', f'val_{self.data_path}.npy', f'test_{self.data_path}.npy'),
            # Pattern without .npy extension in data_path
            (f'{self.data_path}train.npy', f'{self.data_path}val.npy', f'{self.data_path}test.npy'),
        ]
        
        train_file = val_file = test_file = None
        
        # Try to find the files
        for train_name, val_name, test_name in possible_patterns:
            train_path = os.path.join(self.root_path, train_name)
            val_path = os.path.join(self.root_path, val_name)
            test_path = os.path.join(self.root_path, test_name)
            
            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                train_file, val_file, test_file = train_path, val_path, test_path
                print(f"Found data files:")
                print(f"  Train: {train_file}")
                print(f"  Val: {val_file}")
                print(f"  Test: {test_file}")
                break
        
        if train_file is None:
            # List available files to help debug
            if os.path.exists(self.root_path):
                files = os.listdir(self.root_path)
                print(f"Available files in {self.root_path}: {files}")
                print(f"Tried these patterns:")
                for pattern in possible_patterns:
                    print(f"  {pattern}")
            raise FileNotFoundError(f"Could not find train/val/test data files in {self.root_path}")
        
        # Load the data files (exactly like Dataset_Neuro)
        try:
            train_data = np.load(train_file)
            val_data = np.load(val_file)
            test_data = np.load(test_file)
            print(f"Loaded data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        except Exception as e:
            print(f"Error loading numpy files: {e}")
            raise

        # Reshape for scaling (exactly like Dataset_Neuro)
        train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        val_data_reshaped = val_data.reshape(-1, val_data.shape[-1])
        test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])

        if self.scale:
            self.scaler.fit(train_data_reshaped)
            train_data_scaled = self.scaler.transform(train_data_reshaped)
            val_data_scaled = self.scaler.transform(val_data_reshaped)
            test_data_scaled = self.scaler.transform(test_data_reshaped)
        else:
            train_data_scaled = train_data_reshaped
            val_data_scaled = val_data_reshaped
            test_data_scaled = test_data_reshaped

        # Reshape back to original shape (exactly like Dataset_Neuro)
        train_scaled_orig_shape = train_data_scaled.reshape(train_data.shape)
        val_scaled_orig_shape = val_data_scaled.reshape(val_data.shape)
        test_scaled_orig_shape = test_data_scaled.reshape(test_data.shape)

        # Use the appropriate split based on set_type (exactly like Dataset_Neuro)
        if self.set_type == 0:  # TRAIN
            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
            self.data_x = train_x
            self.data_y = train_y

        elif self.set_type == 1:  # VAL
            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
            self.data_x = val_x
            self.data_y = val_y

        elif self.set_type == 2:  # TEST
             test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
             self.data_x = test_x
             self.data_y = test_y

    def make_full_x_y_data(self, array):
        """Create sequences exactly like Dataset_Neuro"""
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Saugeen_Web(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

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

        data_x = np.load(os.path.join(self.root_path, 'all_x_original.npy'))
        data_y = np.load(os.path.join(self.root_path, 'all_y_original.npy'))

        # sensors were already last
        data_x_sensors_last = data_x
        data_y_sensors_last = data_y

        data_x_reshaped = data_x_sensors_last.reshape(-1, data_x_sensors_last.shape[-1])
        data_y_reshaped = data_y_sensors_last.reshape(-1, data_y_sensors_last.shape[-1])

        if self.scale:
            self.scaler.fit(data_x_reshaped)
            data_x_scaled = self.scaler.transform(data_x_reshaped)
            data_y_scaled = self.scaler.transform(data_y_reshaped)

        data_x_scaled_orig_shape = data_x_scaled.reshape(data_x_sensors_last.shape)
        data_y_scaled_orig_shape = data_y_scaled.reshape(data_y_sensors_last.shape)

        self.data_x = data_x_scaled_orig_shape
        self.data_y = data_y_scaled_orig_shape

        print(self.set_type, len(self.data_x), len(self.data_y), self.data_x[0].shape, self.data_y[0].shape)

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        pdb.set_trace()
        return self.scaler.inverse_transform(data)