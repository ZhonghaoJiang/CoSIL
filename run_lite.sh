export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("openai/qwen2.5-7b-instruct")
threads=200
output_prefix="results/swe-bench-lite"

for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/CoSIL_localize_file.py --file_level \
                               --output_folder "${output_prefix}/file_level_${model_tag}" \
                               --num_threads ${threads} \
                               --model "${model}" \
                               --skip_existing
done

for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/CoSIL_localize_func.py \
    --output_folder "${output_prefix}/func_level_${model_tag}" \
    --loc_file "${output_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.85 \
    --model "${model}" \
    --skip_existing \
    --num_threads ${threads}
done




