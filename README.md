# Issue Localization via LLM-Driven Iterative Code Graph Searching

## Environment Setup

```shell
conda create -n cosil python=3.11
conda activate cosil
pip install -r requirments.txt
```

Model calls are routed through [LiteLLM](https://docs.litellm.ai/). Configure credentials and API endpoints with environment variables before running the scripts. For an OpenAI-compatible endpoint, for example:

```shell
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_API_BASE="https://<your-endpoint>/v1"
```

For other providers, use the environment variables supported by LiteLLM and pass the corresponding LiteLLM model name to `--model`.

## How to run?

### Preparation

Generate the repository structure by running:

```shell
python get_lite_structure.py      # For SWE-Bench Lite
python get_verified_structure.py  # For SWE-Bench Verified
```

To avoid regenerating repository structure files repeatedly, you can use the cache provided by the Agentless Team. [Download here](https://github.com/OpenAutoCoder/Agentless/releases/tag/v1.5.0).

Then export the repository-structure location before running localization scripts:

```shell
export PROJECT_FILE_LOC="<path to your repo structures>"
```

You can set this directly in `run_lite.sh`, `run_verified.sh`, `patch_gen.sh`, `ablation.sh`, and `sample.sh`, or export it in your shell.


### RQ1: Effectiveness

To reproduce RQ1 results for SWE-bench Lite/Verified:

```shell
bash run_lite.sh
bash run_verified.sh
```

Results are stored in the `results` folder.

### RQ2: Ablation

```shell
bash ablation.sh
```

### RQ3: Application

```shell
bash patch_gen.sh
```

Then use the official SWE-bench evaluation method to evaluate the generated patches.

### RQ4: Generalizability

```shell
bash sample.sh
```

### Evaluation

Evaluate localization results on SWE-bench Lite or SWE-bench Verified with:

```shell
cd evaluation
python FLEvalNew.py --dataset ["lite"/"verified"] --loc_file ["path to your localization results"]
```

## Acknowledgement

This repository is partially based on OpenAutoCoder/Agentless.

* [Agentless](https://github.com/OpenAutoCoder/Agentless/tree/main)
* [SWE-Bench](https://github.com/swe-bench/SWE-bench.git)
* [OrcaLoca](https://github.com/fishmingyu/OrcaLoca)
