# Issue Localization via LLM-Driven Iteritive Code Graph Searching

In the experiment, we use `afl` to represent our approach, CoSIL.

## Environment Setup

```shell
conda create -n cosil python=3.11
pip install -r requirements.txt
```

## How to run?
### Preparation
To reproduce the full SWE-bench lite/verified experiments, you should first set up your API key 
in `afl/util/api_requests.py`:

```python
client = openai.OpenAI(api_key="sk-xxxx", base_url="https://xxx/v1")
```

Then, you should generate the repository structure by running the following command:
```shell
python get_lite_structure.py # For SWE-Bench Lite
python get_verified_structure.py # For SWE-Bench Verified
```
To avoid regenerating the repository structure files repeatedly,
you can use the cache provided by Agentless Team. [Download Here!](https://github.com/OpenAutoCoder/Agentless/releases/tag/v1.5.0)


After that, you should export the following environment variables in `run_lite.sh`, `run_verified.sh`, `patch_gen.sh` and `ablation.sh` at line 2:
```shell
export PROJECT_FILE_LOC=<path to your repo structures>
```

### RQ1: Effectiveness
To reproduce RQ1's results, you can run the following command to reproduce the full SWE-bench lite/verified experiments:

```shell
bash run_lite.sh
bash run_verified.sh
```
And the results will be stored in `results` folder.

### RQ2: Ablation
To reproduce RQ2's results, you can run the following command.

```shell
bash ablation.sh
```

### RQ3: Application
To reproduce RQ3's results, you can run the following command.

```shell
bash patch_gen.sh
```
And then you can use the official evaluation method to evaluate the generated patches on SWE-Bench.

### RQ4: Generalizbility
To reproduce RQ4's results, you can run the following command.

```shell
bash sample.sh
``` 
### Evaluation
You can use the following command to evaluate the localization results on SWE-bench-Lite or SWE-Bench-Verified.

```shell
cd evaluation
python FLEvalNew.py --dataset ["lite"/"verified"] --loc_file ["path to your localization results"]
```

## Acknowledgement

This repository is partially based on OpenAutoCoder/Agentless.
* [Agentless](https://github.com/OpenAutoCoder/Agentless/tree/main)
* [SWE-Bench](https://github.com/swe-bench/SWE-bench.git)
* [OrcaLoca](https://github.com/fishmingyu/OrcaLoca)


