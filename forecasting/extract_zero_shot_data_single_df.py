import argparse
import numpy as np
import os
import torch
from data_provider.data_factory_whole import data_provider_whole
from lib.models.revin import RevIN


class ExtractDataWhole:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        # For whole data processing, enc_in should be determined from the data
        self.enc_in = args.enc_in if hasattr(args, 'enc_in') else None
        self.revin_layer = None  # Will be initialized after we know the number of features

    def _get_data(self):
        # Get all data without splitting using whole data provider
        data_set, data_loader = data_provider_whole(self.args, flag='test')
        return data_set, data_loader

    def process_data(self, loader, vqvae_model):
        original_all = []
        code_ids_all = []
        reverted_all = []

        # Check if loader is empty
        if len(loader) == 0:
            print(f"ERROR: Data loader is empty! No data found.")
            print(f"Please check:")
            print(f"  - Data path: {self.args.root_path}")
            print(f"  - Data file: {self.args.data_path}")
            print(f"  - Make sure the data files exist and are accessible")
            return None

        for i, (batch_data, _, _, _) in enumerate(loader):
            if i == 0:
                print(f"Processing {len(loader)} batches of whole data...")
                num_sensors = batch_data.shape[-1]
                print(f"Found {num_sensors} sensors")
                print(f"Batch shape: {batch_data.shape}")
                
                # Initialize RevIN layer now that we know the number of features
                if self.revin_layer is None:
                    self.revin_layer = RevIN(num_features=num_sensors, affine=False, subtract_last=False)
                    print(f"Initialized RevIN layer with {num_sensors} features")

            original_all.append(batch_data)

            batch_data = batch_data.float().to(self.device)

            # data going into revin should have dim:[bs x time x nvars]
            data_in_revin_space = self.revin_layer(batch_data, "norm")

            # expects time to be dim [bs x nvars x time]
            codes, code_ids, codebook = time2codes(
                data_in_revin_space.permute(0, 2, 1), 
                self.args.compression_factor, 
                vqvae_model.encoder, 
                vqvae_model.vq
            )

            code_ids_all.append(np.array(code_ids.detach().cpu()))

            # expects code to be dim [bs x nvars x compressed_time]
            predictions_revin_space, predictions_original_space = codes2time(
                code_ids, 
                codebook, 
                self.args.compression_factor, 
                vqvae_model.decoder, 
                self.revin_layer
            )

            reverted_all.append(np.array(predictions_original_space.detach().cpu()))

        # Check if we have any data to process
        if len(original_all) == 0:
            print("ERROR: No batches were processed!")
            return None

        original_arr = np.concatenate(original_all, axis=0)
        code_ids_all_arr = np.concatenate(code_ids_all, axis=0)
        reverted_all_arr = np.concatenate(reverted_all, axis=0)

        data_dict = {}
        data_dict['original_arr'] = original_arr
        data_dict['code_ids_all_arr'] = np.swapaxes(code_ids_all_arr, 1, 2)  # order will be [bs x compressed_time x sensors]
        data_dict['reverted_all_arr'] = reverted_all_arr
        data_dict['codebook'] = np.array(codebook.detach().cpu())

        # Verify sensor dimensions
        if not all(arr.shape[-1] == num_sensors for arr in [
            data_dict['original_arr'],
            data_dict['code_ids_all_arr'],
            data_dict['reverted_all_arr']
        ]):
            raise ValueError("Inconsistent sensor dimensions in processed data")

        print("Data shapes:")
        print(f"Original data: {data_dict['original_arr'].shape}")
        print(f"Code IDs: {data_dict['code_ids_all_arr'].shape}")
        print(f"Reverted data: {data_dict['reverted_all_arr'].shape}")
        print(f"Codebook: {data_dict['codebook'].shape}")

        return data_dict

    def extract_data(self):
        device = 'cuda:' + str(self.args.gpu) if self.args.use_gpu else 'cpu'
        vqvae_model = torch.load(self.args.trained_vqvae_model_path, map_location=device)
        vqvae_model.to(device)
        vqvae_model.eval()

        # Get all data without splitting
        data, loader = self._get_data()

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        print('Processing all data through VQVAE...')
        data_dict = self.process_data(loader, vqvae_model)
        
        if data_dict is not None:
            save_files(self.args.save_path, data_dict)
        else:
            print("Error processing data, no files saved.")


def save_files(path, data_dict):
    """Save all processed data to files"""
    np.save(path + 'original.npy', data_dict['original_arr'])
    np.save(path + 'codes.npy', data_dict['code_ids_all_arr'])
    np.save(path + 'reverted.npy', data_dict['reverted_all_arr'])
    np.save(path + 'codebook.npy', data_dict['codebook'])
    
    print(f"Saved files:")
    print(f"  - original.npy: {data_dict['original_arr'].shape}")
    print(f"  - codes.npy: {data_dict['code_ids_all_arr'].shape}")
    print(f"  - reverted.npy: {data_dict['reverted_all_arr'].shape}")
    print(f"  - codebook.npy: {data_dict['codebook'].shape}")


def time2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    '''
    Args:
        revin_data: [bs x nvars x pred_len or seq_len]
        compression_factor: int
        vqvae_model: trained vqvae model

    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time]
        embedding_weight: [num_code_words, code_dim]
    '''
    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]
    compressed_time = int(T / compression_factor)

    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, T)
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim, compressed_time)
        code_ids = encoding_indices.view(bs, nvar, compressed_time)

    return codes, code_ids, embedding_weight


def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    '''
    Args:
        code_ids: [bs x nvars x compressed_pred_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_model: trained vqvae model
    Returns:
        predictions_revin_space: [bs x original_time_len x nvars]
        predictions_original_space: [bs x original_time_len x nvars]
    '''
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device), 1)
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)
        quantized_swaped = torch.swapaxes(quantized, 1, 2)
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1, 2)
        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')

    return predictions_revin_space, predictions_original_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VQVAE codes for zero-shot forecasting - Whole data processing')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, help='dataset type')
    parser.add_argument('--root_path', type=str, required=True, help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')

    # forecasting task - not needed for whole data processing
    parser.add_argument('--seq_len', type=int, default=None, help='input sequence length (not used for whole data processing)')
    parser.add_argument('--pred_len', type=int, default=None, help='prediction sequence length (not used for whole data processing)')
    parser.add_argument('--enc_in', type=int, default=64, help='encoder input size (number of channels, auto-detected from data)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Save Location
    parser.add_argument('--save_path', type=str, required=True,
                        help='folder ending in / where we want to save the processed data')
    parser.add_argument('--trained_vqvae_model_path', type=str, required=True,
                        help='path to the trained VQVAE model')
    parser.add_argument('--compression_factor', type=int, required=True, help='compression factor for VQVAE')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--classifiy_or_forecast', type=str, default='forecast',
                        help='classify or forecast, options:[classify, forecast]')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    args = parser.parse_args()

    # Set random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)
    print('GPU:', torch.cuda.is_available())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU:', torch.cuda.get_device_name(args.gpu))

    exp = ExtractDataWhole(args)
    exp.extract_data()