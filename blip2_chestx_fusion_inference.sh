python inference/blip2_fusion_inference.py \
--test_csv data/chestx/proc/test.csv \
--image_folder data/chestx/proc/images \
--model_path models/blip2_chestx_fusion_model/epoch10 \
--output_csv predictions/predictions_fusion.csv \
--batch_size 4 \
--max_length 512