from data_provider.data_loader_whole import Dataset_Neuro_Whole, Dataset_Neuro_Whole_Single
from torch.utils.data import DataLoader

data_dict = {
    'neuro_zero_shot': Dataset_Neuro_Whole,  # Process each trial separately
    'neuro_zero_shot_single': Dataset_Neuro_Whole_Single,  # Process as one long sequence
}


def data_provider_whole(args, flag):
    """
    Data provider for whole data processing without windowing.
    
    Args:
        args: Arguments containing data configuration
        flag: Should be 'test' for whole data processing
        
    Returns:
        data_set, data_loader: Dataset and DataLoader for whole data processing
    """
    Data = data_dict[args.data]
    
    # For whole data processing, we typically don't need time encoding
    timeenc = 0 if args.embed != 'timeF' else 1

    # Always use test configuration for whole data processing
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
    freq = args.freq if hasattr(args, 'freq') else 'h'
    
    # Create dataset
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=None,  # No windowing for whole data processing
        features=args.features,
        target=args.target if hasattr(args, 'target') else 'OT',
        timeenc=timeenc,
        freq=freq
    )
    
    print(f"Whole data processing - Dataset size: {len(data_set)}")
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 1,
        drop_last=drop_last
    )
    
    return data_set, data_loader