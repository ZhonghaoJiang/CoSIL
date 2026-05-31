export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Fault Localization
models=("openai/qwen2.5-coder-32b-instruct")
threads=5

module_prefix="results/ablation_module_call_graph"
reflection_prefix="results/ablation_reflection"
func_prefix="results/ablation_func"
round_prefix="results/round"

for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/ablation_module_call_graph.py --file_level \
                               --output_folder "${module_prefix}/file_level_${model_tag}" \
                               --num_threads ${threads} \
                               --model "${model}" \
                               --skip_existing

  python CoSIL/fl/CoSIL_localize_func.py \
  --output_folder "${module_prefix}/func_level_${model_tag}" \
  --loc_file "${module_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
  --output_file "loc_${model_tag}_func.jsonl" \
  --temperature 0.0 \
  --model "${model}" \
  --skip_existing \
  --num_threads ${threads}

done

for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/ablation_reflection.py --file_level \
                               --output_folder "${reflection_prefix}/file_level_${model_tag}" \
                               --num_threads ${threads} \
                               --model "${model}" \
                               --skip_existing

  python CoSIL/fl/CoSIL_localize_func.py \
  --output_folder "${reflection_prefix}/func_level_${model_tag}" \
  --loc_file "${reflection_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
  --output_file "loc_${model_tag}_func.jsonl" \
  --temperature 0.0 \
  --model "${model}" \
  --skip_existing \
  --num_threads ${threads}

done


for model in "${models[@]}"; do
  model_tag=${model//\//_}
  python CoSIL/fl/CoSIL_localize_file.py --file_level \
                             --output_folder "${func_prefix}/file_level_${model_tag}" \
                             --num_threads ${threads} \
                             --model "${model}" \
                             --skip_existing

  python CoSIL/fl/ablation_func.py \
    --output_folder "${func_prefix}/func_level_${model_tag}" \
    --loc_file "${func_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.0 \
    --model "${model}" \
    --skip_existing \
    --num_threads ${threads}
done



for model in "${models[@]}"; do
  model_tag=${model//\//_}
  mkdir -p "${round_prefix}"
  cp -r "${func_prefix}/file_level_${model_tag}" "${round_prefix}/file_level_${model_tag}"

  python CoSIL/fl/CoSIL_localize_func.py \
    --output_folder "${round_prefix}/func_level_${model_tag}_1" \
    --loc_file "${round_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.0 \
    --model "${model}" \
    --skip_existing \
    --max_retry 1 \
    --num_threads ${threads}

  python CoSIL/fl/CoSIL_localize_func.py \
    --output_folder "${round_prefix}/func_level_${model_tag}_3" \
    --loc_file "${round_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.0 \
    --model "${model}" \
    --skip_existing \
    --max_retry 3 \
    --num_threads ${threads}

  python CoSIL/fl/CoSIL_localize_func.py \
    --output_folder "${round_prefix}/func_level_${model_tag}_5" \
    --loc_file "${round_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
    --output_file "loc_${model_tag}_func.jsonl" \
    --temperature 0.0 \
    --model "${model}" \
    --skip_existing \
    --max_retry 5 \
    --num_threads ${threads}

  python CoSIL/fl/CoSIL_localize_func.py \
  --output_folder "${round_prefix}/func_level_${model_tag}_7" \
  --loc_file "${round_prefix}/file_level_${model_tag}/loc_outputs.jsonl" \
  --output_file "loc_${model_tag}_func.jsonl" \
  --temperature 0.0 \
  --model "${model}" \
  --skip_existing \
  --max_retry 7 \
  --num_threads ${threads}
done


