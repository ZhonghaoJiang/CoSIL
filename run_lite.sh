export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("qwen_2.5_7b")
model_names=("qwen2.5-7b-instruct")
backend=("openai")

for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_file.py --file_level \
                               --output_folder "results/swe-bench-lite/file_level_${models[$i]}" \
                               --num_threads 100 \
                               --model "${model_names[$i]}" \
                               --backend "${backend[$i]}" \
                               --skip_existing
done

for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_func.py \
    --output_folder "results/swe-bench-lite/func_level_${models[$i]}" \
    --loc_file "results/swe-bench-lite/file_level/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.85 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --num_threads 100
done

for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_line.py \
    --output_folder "results/swe-bench-lite/line_level_${models[$i]}" \
    --loc_file "results/swe-bench-lite/func_level_${models[$i]}/loc_${models[$i]}_func.jsonl" \
    --output_file "loc_${models[$i]}_line.jsonl" \
    --temperature 0.85 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --num_threads 100
done



