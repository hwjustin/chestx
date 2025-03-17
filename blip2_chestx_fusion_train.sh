export SEED=3

# Baseline model training
python blip2_fusion_train.py \
--train_csv data_split/chestx_split_results.csv \
--report_csv proc/train.csv \
--image_folder proc/images \
--save_path models/blip2_chestx_baseline_model_${SEED} \
--batch_size 1 \
--epochs 10 \
--lr 5e-5 \
--seed ${SEED} \
--max_length 512