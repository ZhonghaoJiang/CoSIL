export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Patch Generation

model_name="qwen2.5-coder-32b-instruct"
backend="openai"
threads=5

sleep 900

methods=("afl" "agentless" "orcaloca")
folders=("loc_to_patch/afl" "loc_to_patch/agentless" "loc_to_patch/orcaloca")
scales=("32b")

for i in "${!methods[@]}"; do
  for j in "${!scales[@]}"; do
      python agentless/fl/localize.py --fine_grain_line_level \
                                  --output_folder "results/line_level" \
                                  --output_file "${methods[i]}_${scales[j]}.jsonl" \
                                  --top_n 3 \
                                  --compress \
                                  --temperature 0.8 \
                                  --num_samples 1 \
                                  --start_file "${folders[$i]}/${methods[i]}_qwen_coder_${scales[j]}_func.jsonl" \
                                  --num_threads ${threads} \
                                  --model "${model_name}" \
                                  --backend "${backend}" \
                                  --skip_existing
  done

done


for i in "${!methods[@]}"; do
  for j in "${!scales[@]}"; do
    python agentless/repair/repair.py --loc_file "results/line_level/${methods[i]}_${scales[j]}.jsonl" \
                                      --output_folder "results/patches/repair_sample_${methods[i]}_${scales[j]}" \
                                      --loc_interval \
                                      --top_n=3 \
                                      --context_window=10 \
                                      --max_samples 20  \
                                      --cot \
                                      --diff_format \
                                      --gen_and_process \
                                      --num_threads ${threads} \
                                      --model "${model_name}" \
                                      --backend "${backend}"
  done
done

folders=("results/patches/repair_sample_afl_32b")
for folder in "${folders[@]}"; do
    run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
    python agentless/repair/rerank.py --patch_folder ${folder}  \
                                    --output_file all_preds_${run_id_prefix}.jsonl \
                                    --num_samples 10 \
                                    --deduplicate
done
