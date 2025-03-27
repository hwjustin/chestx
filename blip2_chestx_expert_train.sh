export SEED=3

# Baseline model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/chestx_split_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_baseline_model \
--batch_size 4 \
--epochs 10 \
--lr 5e-5 \
--seed ${SEED} \
--max_length 512

# # AS category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/AS_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_AS_model \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# # R category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/R_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_R_model \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# T category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/T_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_T_model \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# I category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/I_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_I_model \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512


# U category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/train/U_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_U_model \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

