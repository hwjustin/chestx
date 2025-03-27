export SEED=3

# Baseline model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/chestx_split_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_baseline_model_${SEED} \
--batch_size 4 \
--epochs 10 \
--lr 5e-5 \
--seed ${SEED} \
--max_length 512

# # AS category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/AS_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_AS_model_${SEED} \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# # R category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/R_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_R_model_${SEED} \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# T category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/T_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_T_model_${SEED} \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512

# I category model training
python train/blip2_expert_train.py \
--train_csv data/chestx/split/I_category_results.csv \
--report_csv data/chestx/proc/train.csv \
--image_folder data/chestx/proc/images \
--save_path models/blip2_chestx_I_model_${SEED} \
--batch_size 4 \
--epochs 10 \
--lr 1e-4 \
--seed ${SEED} \
--max_length 512
