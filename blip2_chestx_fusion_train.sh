export SEED=3

# Baseline model training
python train/blip2_fusion_train.py \
--train_csv data/chestx/split/chestx_split_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_fusion_model \
--batch_size 4 \
--epochs 10 \
--lr 5e-5 \
--seed ${SEED} \
--max_length 512