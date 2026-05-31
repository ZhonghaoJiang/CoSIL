export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("openai/gpt-4o-2024-08-06")
threads=5
dataset_name="SWE-bench_Lite_sampled"
output_prefix="results/sample-lite"

# for model in "${models[@]}"; do
#   model_tag=${model//\//_}
#   python CoSIL/fl/CoSIL_localize_file.py --file_level \
#                                --output_folder "${output_prefix}/file_level_${model_tag}" \
#                                --num_threads ${threads} \
#                                --model "${model}" \
#                                --dataset ${dataset_name} \
#                                --skip_existing

# done


for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/CoSIL_localize_func.py \
    --output_folder "${output_prefix}/func_level_${model_tag}" \
    --loc_file "${output_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.0 \
    --model "${model}" \
    --dataset ${dataset_name} \
    --skip_existing \
    --num_threads ${threads}
done





