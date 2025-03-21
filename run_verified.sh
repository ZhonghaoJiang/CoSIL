export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("qwen2.5-14b")
model_names=("qwen-coder-14b")
backend=("openai")
threads=200

for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_file.py --file_level \
                               --output_folder "results/swe-bench-verified/file_level_${models[$i]}" \
                               --num_threads ${threads} \
                               --model "${model_names[$i]}" \
                               --backend "${backend[$i]}" \
                               --dataset "princeton-nlp/SWE-bench_Verified" \
                               --skip_existing

done


for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_func.py \
    --output_folder "results/swe-bench-verified/func_level_${models[$i]}" \
    --loc_file "results/swe-bench-verified/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func_2.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --dataset "princeton-nlp/SWE-bench_Verified" \
    --skip_existing \
    --num_threads ${threads}
done





