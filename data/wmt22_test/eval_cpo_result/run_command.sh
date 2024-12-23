cd /mnt/data/haiye/llms/metricx-23/metricx

root_to_output=/mnt/data/haiye/ensemble_inference/ensemble_inference/data/wmt22_test/eval_cpo_result
metric_name=metricx-23-xxl

input_file=$root_to_output/$1
output_file=$root_to_output/$2


CUDA_VISIBLE_DEVICES=$3 python -m metricx23.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path  /mnt/data/haiye/llms/metricx-23/models--google--metricx-23-xxl-v2p0/snapshots/599c176bdfb5fb56689ab368a0522dc0513f86da \
  --max_input_length 1024 \
  --batch_size 1 \
  --input_file $input_file \
  --output_file $output_file



