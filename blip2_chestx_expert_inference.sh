export SEED=3

# Inference with baseline model
python inference/blip2_expert_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_baseline_model/epoch10 \
--output_csv predictions/predictions_baseline.csv \
--batch_size 4 \
--max_length 512


# Inference with AS category model
python inference/blip2_expert_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_AS_model/epoch10 \
--output_csv predictions/predictions_AS.csv \
--batch_size 4 \
--max_length 512


# Inference with R category model
python inference/blip2_expert_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_R_model/epoch10 \
--output_csv predictions/predictions_R.csv \
--batch_size 4 \
--max_length 512

# Inference with T category model
python inference/blip2_expert_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_T_model/epoch10 \
--output_csv predictions/predictions_T.csv \
--batch_size 4 \
--max_length 512

# Inference with I category model
python inference/blip2_expert_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_I_model/epoch10 \
--output_csv predictions/predictions_I.csv \
--batch_size 4 \
--max_length 512




