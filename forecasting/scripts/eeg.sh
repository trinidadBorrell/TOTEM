gpu=0
random_seed=2021
data_path_name=neuro_zero_shot
model_id_name=neuro_zero_shot
data_name=neuro_zero_shot
seq_len=85
pred_len=69
type_pretrained=forecast
sub=AA074

python forecasting/extract_zero_shot_data_single_df.py \
  --data neuro_zero_shot \
  --root_path "/data/project/eeg_foundation/data/data_local_global/lg_data_processed/sub-${sub}/" \
  --data_path neuro_zero_shot \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 64 \
  --label_len 0 \
  --gpu $gpu \
  --save_path "/home/triniborrell/home/projects/TOTEM/forecasting/data/all_vqvae_extracted/nice/Tin${seq_len}_Tout${pred_len}_${type_pretrained}_${sub}/" \
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