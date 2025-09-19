export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("gpt")
model_names=("gpt-4o-2024-08-06")
backend=("openai")
threads=5
dataset_name="SWE-bench_Lite_sampled"

# for i in "${!models[@]}"; do
#   python afl/fl/AFL_localize_file.py --file_level \
#                                --output_folder "results/sample-lite/file_level_${models[$i]}" \
#                                --num_threads ${threads} \
#                                --model "${model_names[$i]}" \
#                                --backend "${backend[$i]}" \
#                                --dataset ${dataset_name} \
#                                --skip_existing

# done


for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_func.py \
    --output_folder "results/sample-lite/func_level_${models[$i]}" \
    --loc_file "results/sample-lite/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --dataset ${dataset_name} \
    --skip_existing \
    --num_threads ${threads}
done





