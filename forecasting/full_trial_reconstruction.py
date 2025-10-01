#!/usr/bin/env python3
"""
Full Trial Reconstruction Script

This script reconstructs entire trials (all timesteps) using VQVAE 
for quality assessment, without the forecasting pipeline's X/Y splitting.
"""

import argparse
import numpy as np
import os
import torch
from data_provider.data_factory import data_provider
from lib.models.revin import RevIN

def time2codes_full(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    """
    Compress full sequences without X/Y splitting
    
    Args:
        revin_data: [bs x nvars x full_seq_len] 
        compression_factor: int
        vqvae_encoder: encoder model
        vqvae_quantizer: quantizer model
    
    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time] 
        embedding_weight: [num_code_words, code_dim]
    """
    bs = revin_data.shape[0]
    nvar = revin_data.shape[1] 
    T = revin_data.shape[2]
    compressed_time = int(T / compression_factor)
    
    with torch.no_grad():
        flat_revin = revin_data.reshape(-1, T)  # [bs * nvars, T]
        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)
        
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, nvar, code_dim, compressed_time)
        code_ids = encoding_indices.view(bs, nvar, compressed_time)
    
    return codes, code_ids, embedding_weight

def codes2time_full(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    """
    Reconstruct full sequences from codes
    
    Args:
        code_ids: [bs x nvars x compressed_seq_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_decoder: decoder model
        revin_layer: RevIN layer for denormalization
        
    Returns:
        predictions_revin_space: [bs x full_seq_len x nvars]
        predictions_original_space: [bs x full_seq_len x nvars]
    """
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)
    
    with torch.no_grad():
        # Scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device), 1)
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)
        quantized_swapped = torch.swapaxes(quantized, 1, 2)  # [bs * nvars, code_dim, compressed_len]
        
        prediction_recon = vqvae_decoder(quantized_swapped.to(device), compression_factor)
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1, 2)  # [bs x seq_len x nvars]
        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')
    
    return predictions_revin_space, predictions_original_space

class FullTrialReconstructor:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
        
    def reconstruct_full_trials(self):
        """Reconstruct entire trials without X/Y splitting"""
        device = 'cuda:' + str(self.args.gpu) if self.args.use_gpu else 'cpu'
        vqvae_model = torch.load(self.args.trained_vqvae_model_path, map_location=device)
        vqvae_model.to(device)
        vqvae_model.eval()
        
        # Create temporary args for full sequence loading
        temp_args = argparse.Namespace(**vars(self.args))
        temp_args.seq_len = self.args.full_seq_len  # Use full trial length
        temp_args.pred_len = 0  # No prediction, just reconstruction
        temp_args.label_len = 0
        
        # Load data with full sequence length
        from data_provider.data_loader import Dataset_Neuro_Custom
        
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
            
        for split in ['train', 'test', 'val']:
            print(f'\n============= {split.upper()} =============')
            
            # Load raw data directly
            data_files = {
                'train': f'{split}_data.npy',
                'test': f'{split}_data.npy', 
                'val': f'{split}_data.npy'
            }
            
            try:
                raw_data = np.load(os.path.join(self.args.root_path, data_files[split]))
                print(f"Loaded {split} data: {raw_data.shape}")
                
                # Process in batches to avoid memory issues
                batch_size = min(32, raw_data.shape[0])
                n_batches = (raw_data.shape[0] + batch_size - 1) // batch_size
                
                all_original = []
                all_reconstructed = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, raw_data.shape[0])
                    batch_data = raw_data[start_idx:end_idx]
                    
                    # Convert to torch tensor and move to device
                    batch_tensor = torch.FloatTensor(batch_data).to(device)
                    
                    # Apply RevIN normalization [bs x time x nvars] -> [bs x time x nvars]
                    batch_normalized = self.revin_layer(batch_tensor, "norm")
                    
                    # Compress to codes [bs x nvars x time]
                    batch_transposed = batch_normalized.permute(0, 2, 1)
                    codes, code_ids, codebook = time2codes_full(
                        batch_transposed, self.args.compression_factor, 
                        vqvae_model.encoder, vqvae_model.vq
                    )
                    
                    # Reconstruct from codes
                    recon_revin_space, recon_original_space = codes2time_full(
                        code_ids, codebook, self.args.compression_factor,
                        vqvae_model.decoder, self.revin_layer
                    )
                    
                    # Store results
                    all_original.append(batch_data)
                    all_reconstructed.append(recon_original_space.detach().cpu().numpy())
                    
                    print(f"Processed batch {batch_idx + 1}/{n_batches}")
                
                # Concatenate all batches
                original_full = np.concatenate(all_original, axis=0)
                reconstructed_full = np.concatenate(all_reconstructed, axis=0)
                
                print(f"Final shapes - Original: {original_full.shape}, Reconstructed: {reconstructed_full.shape}")
                
                # Save results
                np.save(os.path.join(self.args.save_path, f"{split}_original_full.npy"), original_full)
                np.save(os.path.join(self.args.save_path, f"{split}_reconstructed_full.npy"), reconstructed_full)
                
                # Save codebook (once)
                if split == 'train':
                    np.save(os.path.join(self.args.save_path, "codebook_full.npy"), codebook.detach().cpu().numpy())
                
                print(f"Saved {split} full trial reconstructions")
                
            except Exception as e:
                print(f"Error processing {split}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Trial Reconstruction for Quality Assessment')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='neuro_custom', help='dataset type')
    parser.add_argument('--root_path', type=str, required=True, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='neuro_custom', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    
    # Model parameters
    parser.add_argument('--enc_in', type=int, default=64, help='encoder input size')
    parser.add_argument('--full_seq_len', type=int, default=154, help='full sequence length to reconstruct')
    
    # VQVAE parameters
    parser.add_argument('--trained_vqvae_model_path', type=str, required=True, help='path to trained VQVAE model')
    parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
    
    # System parameters
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--save_path', type=str, required=True, help='save path for results')
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Setup GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    print('Args in experiment:')
    print(args)
    print('GPU available:', torch.cuda.is_available())
    
    # Run reconstruction
    reconstructor = FullTrialReconstructor(args)
    reconstructor.reconstruct_full_trials() 