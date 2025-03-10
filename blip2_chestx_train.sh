export SEED=3

# Baseline model training
python blip2_train.py \
--train_csv data_split/chestx_split_results.csv \
--report_csv proc/train.csv \
--image_folder proc/images \
--save_path models/blip2_chestx_baseline_model_${SEED} \
--batch_size 8 \
--epochs 10 \
--lr 5e-5 \
--seed ${SEED} \
--max_length 512

# # AS category model training
# python blip2_train.py \
# --train_csv data_split/AS_category_results.csv \
# --report_csv proc/train.csv \
# --image_folder proc/images \
# --load_from_ckpt models/blip2_chestx_baseline_model_${SEED} \
# --save_path models/blip2_chestx_AS_model_${SEED} \
# --batch_size 8 \
# --epochs 10 \
# --lr 5e-5 \
# --seed ${SEED} \
# --max_length 512

# # R category model training
# python blip2_train.py \
# --train_csv data_split/R_category_results.csv \
# --report_csv proc/train.csv \
# --image_folder proc/images \
# --load_from_ckpt models/blip2_chestx_baseline_model_${SEED} \
# --save_path models/blip2_chestx_R_model_${SEED} \
# --batch_size 8 \
# --epochs 10 \
# --lr 5e-5 \
# --seed ${SEED} \
# --max_length 512

# # U category model training
# python blip2_train.py \
# --train_csv data_split/U_category_results.csv \
# --report_csv proc/train.csv \
# --image_folder proc/images \
# --load_from_ckpt models/blip2_chestx_baseline_model_${SEED} \
# --save_path models/blip2_chestx_U_model_${SEED} \
# --batch_size 8 \
# --epochs 10 \
# --lr 5e-5 \
# --seed ${SEED} \
# --max_length 512
