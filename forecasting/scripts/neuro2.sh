
gpu=0
random_seed=2021
data_path_name=neuro_custom
model_id_name=neuro_custom
data_name=neuro_custom
seq_len=77
pred_len=77
#for pred_len in 96 192 336 720
#do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "/home/triniborrell/home/data/nice_data_processed/sub-AA048/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 64 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/nice/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/home/triniborrell/home/projects/TOTEM/forecasting/generalist/checkpoints/final_model.pth'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

'''
gpu=0
Tin=96
Tout=96 # 192 336 720 # uncomment when want all forecasting lengths
seed=2021
dt=neuro2
python forecasting/generalist_eval.py \
  --data-type $dt \
  --data_path "forecasting/data/all_vqvae_extracted/"$dt"/Tin"$Tin"_Tout"$Tout"/" \
  --model_load_path "forecasting/saved_models/all/forecaster_checkpoints/all_Tin"$Tin"_Tout"$Tout"_seed"$seed"/"\
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu
done
'''