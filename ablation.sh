export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("qwen2.5-32b")
model_names=("qwen2.5-coder-32b-instruct")
backend=("openai")
threads=5

for i in "${!models[@]}"; do
  python afl/fl/ablation_file.py --file_level \
                               --output_folder "results/ablation_file/file_level_${models[$i]}" \
                               --num_threads ${threads} \
                               --model "${model_names[$i]}" \
                               --backend "${backend[$i]}" \
                               --skip_existing

  python afl/fl/AFL_localize_func.py \
  --output_folder "results/ablation_file/func_level_${models[$i]}" \
  --loc_file "results/ablation_file/file_level_${models[$i]}/loc_outputs.jsonl" \
  --output_file "loc_${models[$i]}_func.jsonl" \
  --temperature 0.0 \
  --model "${model_names[$i]}" \
  --backend "${backend[$i]}" \
  --skip_existing \
  --num_threads ${threads}

done


for i in "${!models[@]}"; do
  python afl/fl/AFL_localize_file.py --file_level \
                             --output_folder "results/ablation_func/file_level_${models[$i]}" \
                             --num_threads ${threads} \
                             --model "${model_names[$i]}" \
                             --backend "${backend[$i]}" \
                             --skip_existing

  python afl/fl/ablation_func.py \
    --output_folder "results/ablation_func/func_level_${models[$i]}" \
    --loc_file "results/ablation_func/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --num_threads ${threads}
done



for i in "${!models[@]}"; do
  mkdir results/round
  cp -r "results/ablation_func/file_level_${models[$i]}" "results/round/file_level_${models[$i]}"

  python afl/fl/AFL_localize_func.py \
    --output_folder "results/round/func_level_${models[$i]}_1" \
    --loc_file "results/round/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --max_retry 1 \
    --num_threads ${threads}

  python afl/fl/AFL_localize_func.py \
    --output_folder "results/round/func_level_${models[$i]}_3" \
    --loc_file "results/round/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --max_retry 3 \
    --num_threads ${threads}

  python afl/fl/AFL_localize_func.py \
    --output_folder "results/round/func_level_${models[$i]}_5" \
    --loc_file "results/round/file_level_${models[$i]}/loc_outputs.jsonl" \
    --output_file "loc_${models[$i]}_func.jsonl" \
    --temperature 0.0 \
    --model "${model_names[$i]}" \
    --backend "${backend[$i]}" \
    --skip_existing \
    --max_retry 5 \
    --num_threads ${threads}

  python afl/fl/AFL_localize_func.py \
  --output_folder "results/round/func_level_${models[$i]}_5" \
  --loc_file "results/round/file_level_${models[$i]}/loc_outputs.jsonl" \
  --output_file "loc_${models[$i]}_func.jsonl" \
  --temperature 0.0 \
  --model "${model_names[$i]}" \
  --backend "${backend[$i]}" \
  --skip_existing \
  --max_retry 7 \
  --num_threads ${threads}
done


