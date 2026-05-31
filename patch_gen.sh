export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""
export HF_ENDPOINT=https://hf-mirror.com

# Patch Generation

model="openai/qwen2.5-coder-32b-instruct"
threads=5
dataset="princeton-nlp/SWE-bench_Lite"
line_level_prefix="results/line_level"
patch_prefix="results/patches"
generated_tests_prefix="results/generated_tests"


methods=("CoSIL" "agentless" "orcaloca" "locagent")
folders=("loc_to_patch/CoSIL" "loc_to_patch/agentless" "loc_to_patch/orcaloca" "loc_to_patch/locagent")
scales=("32b")

for i in "${!methods[@]}"; do
  for j in "${!scales[@]}"; do
      python agentless/fl/localize.py --fine_grain_line_level \
                                  --output_folder "${line_level_prefix}" \
                                  --output_file "${methods[i]}_${scales[j]}.jsonl" \
                                  --top_n 3 \
                                  --compress \
                                  --temperature 0.8 \
                                  --num_samples 1 \
                                  --start_file "${folders[$i]}/${methods[i]}_qwen_coder_${scales[j]}_func.jsonl" \
                                  --num_threads ${threads} \
                                  --model "${model}" \
                                  --dataset "${dataset}" \
                                  --skip_existing
  done

done


for i in "${!methods[@]}"; do
  for j in "${!scales[@]}"; do
    python agentless/repair/repair.py --loc_file "${line_level_prefix}/${methods[i]}_${scales[j]}.jsonl" \
                                      --output_folder "${patch_prefix}/repair_sample_${methods[i]}_${scales[j]}" \
                                      --loc_interval \
                                      --top_n=3 \
                                      --context_window=10 \
                                      --max_samples 20  \
                                      --cot \
                                      --diff_format \
                                      --line_level \
                                      --gen_and_process \
                                      --dataset "${dataset}" \
                                      --num_threads ${threads} \
                                      --model "${model}"
  done
done

python agentless/test/run_regression_tests.py --run_id generate_regression_tests \
                                             --dataset "${dataset}" \
                                             --output_file ${generated_tests_prefix}/passing_tests.jsonl


python agentless/test/select_regression_tests.py --passing_tests ${generated_tests_prefix}/passing_tests.jsonl \
                                               --output_folder ${generated_tests_prefix}/select_regression \
                                               --dataset "${dataset}"  \
                                               --model "${model}"
                                               
python agentless/test/generate_reproduction_tests.py --max_samples 40 \
                                                   --output_folder ${generated_tests_prefix}/reproduction_test_samples \
                                                   --dataset "${dataset}" \
                                                   --num_threads ${threads} \
                                                   --model "${model}"

for st in {0..36..4}; do   en=$((st + 3));
      echo "Processing ${st} to ${en}";
      for num in $(seq $st $en); do
          echo "Processing ${num}";
          python3 agentless/test/run_reproduction_tests.py --run_id="reproduction_test_generation_filter_sample_${num}" \
                                                          --dataset "${dataset}"  \
                                                          --test_jsonl="${generated_tests_prefix}/reproduction_test_samples/output_${num}_processed_reproduction_test.jsonl" \
                                                          --num_workers 20 \
                                                          --testing
done & done
wait

python agentless/test/generate_reproduction_tests.py --max_samples 40 \
                                                   --output_folder ${generated_tests_prefix}/reproduction_test_samples \
                                                   --output_file reproduction_tests.jsonl \
                                                   --dataset "${dataset}" \
                                                   --select

folders=(
     "results/patches/repair_sample_CoSIL_32b"
         "results/patches/repair_sample_agentless_32b"
         "results/patches/repair_sample_orcaloca_32b"
         "results/patches/repair_sample_locagent_32b"
         "results/patches_gpt/repair_sample_CoSIL_32b"
         "results/patches_gpt/repair_sample_agentless_32b"
         "results/patches_gpt/repair_sample_orcaloca_32b"
         "results/patches_gpt/repair_sample_locagent_32b"
        )


for folder in "${folders[@]}"; do
    for num in {0..9..1}; do
        run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
        output_file="${folder}/output_${num}_reproduction_test_results.jsonl"

        # 检查目标文件是否存在
        if [[ -f "$output_file" ]]; then
            echo "文件 ${output_file} 已存在，跳过此项。"
            continue
        fi

        python agentless/test/run_reproduction_tests.py --test_jsonl ${generated_tests_prefix}/reproduction_test_samples/reproduction_tests.jsonl \
                                                        --dataset "${dataset}" \
                                                        --predictions_path="${folder}/output_${num}_processed.jsonl" \
                                                        --run_id="${run_id_prefix}_reproduction_${num}" --num_workers 50
    done
done

for folder in "${folders[@]}"; do
    for num in {0..9..1}; do
        run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
        output_file="${folder}/output_${num}_regression_test_results.jsonl"

        # 检查目标文件是否存在
        if [[ -f "$output_file" ]]; then
            echo "文件 ${output_file} 已存在，跳过此项。"
            continue
        fi

        python agentless/test/run_regression_tests.py --regression_tests ${generated_tests_prefix}/select_regression/output.jsonl \
                                                      --predictions_path="${folder}/output_${num}_processed.jsonl" \
                                                      --dataset "${dataset}" \
                                                      --run_id="${run_id_prefix}_regression_${num}" --num_workers 50
    done
done

for folder in "${folders[@]}"; do
    run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
    python agentless/repair/rerank.py --patch_folder ${folder}  \
                                    --output_file new_${run_id_prefix}.jsonl \
                                    --num_samples 10 \
                                    --deduplicate \
                                    --regression \
                                    --reproduction
done

for folder in "${folders[@]}"; do
    run_id_prefix=$(basename ${folder%/*})-$(basename $folder)
    python3 -m swebench.harness.run_evaluation \
                --dataset_name princeton-nlp/SWE-bench_Verified \
                --predictions_path "new_${run_id_prefix}.jsonl" \
                --max_workers 100 \
                --cache_level instance \
                --run_id "${run_id_prefix}_new"
done
