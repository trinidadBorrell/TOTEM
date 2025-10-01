gpu=0
random_seed=2025
data_path_name=neuro_zero_shot
model_id_name=neuro_zero_shot
data_name=neuro_zero_shot
seq_len=85
pred_len=69
type_pretrained=forecast
sub=001

python forecasting/extract_zero_shot_data_single_df.py \
  --data neuro_zero_shot \
  --root_path "/data/project/eeg_foundation/data/processed_nice_data_256/sub-001/ses-01" \
  --data_path neuro_zero_shot \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 256 \
  --label_len 0 \
  --gpu $gpu \
  --save_path "/data/project/eeg_foundation/data/zero_shot_data/sanity_check/" \
  --trained_vqvae_model_path '/home/triniborrell/home/projects/TOTEM/forecasting/pretrained/forecasting/checkpoints/final_model.pth' \
  --compression_factor 4
 --classifiy_or_forecast "forecast"
done

"""
gpu=0
Tin=85
Tout=69 # 192 336 720 # uncomment when want all forecasting lengths
seed=2021
dt=nice
python forecasting/generalist_eval.py \
  --data-type $dt \
  --data_path "forecasting/data/all_vqvae_extracted/"$dt"/Tin"$Tin"_Tout"$Tout"/" \
  --model_load_path "/home/triniborrell/home/projects/TOTEM/forecasting/generalist/checkpoints/" \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu
done
"""