import argparse
import concurrent.futures
import json
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from afl.fl.AFL import AFL
from afl.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
)
from afl.util.utils import (
    load_existing_instance_ids,
    load_json,
    setup_logger,
)
from get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch,
)

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)


def localize_instance(
        bug, args, swe_bench_data, existing_instance_ids
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if bug["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    if PROJECT_FILE_LOC is not None:
        project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
        d = load_json(project_file)
    else:
        # we need to get the project structure directly
        d = get_project_structure_from_scratch(
            bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
        )

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = d["structure"]

    filter_none_python(structure)  # some basic filtering steps

    # filter out test files (unless its pytest)
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)

    # file level localization
    fl = AFL(
        d["instance_id"],
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger
    )

    def load_file_func(output_file, instance_id=None):
        file_locations = []

        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("instance_id") == instance_id:
                    file_locations = data.get("found_files", [])
                    break  # 找到匹配的 instance_id 后退出循环

        return file_locations

    pred_files = load_file_func(args.loc_file, instance_id=instance_id)[: args.top_n]
    # print(pred_files, found_related_locs)
    # 构建字典
    topn_func, func_raw_output, func_traj = fl.ablation_func(file=pred_files, max_retry=args.max_retry)


    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": d["instance_id"],
                    "found_files": pred_files,
                    "found_related_locs": topn_func,
                }
            )
            + "\n"
        )


def localize(args):
    swe_bench_data = load_from_disk("./datasets/SWE-bench_Lite_test")
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in swe_bench_data:
            localize_instance(
                bug, args, swe_bench_data, existing_instance_ids
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_instance,
                    bug,
                    args,
                    swe_bench_data,
                    existing_instance_ids,
                )
                for bug in swe_bench_data
            ]
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(swe_bench_data),
                    colour="MAGENTA",
            ):
                future.result()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs_func.jsonl")
    parser.add_argument("--loc_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--max_retry", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-08-06",
    )
    parser.add_argument(
        "--backend", type=str, default="openai", choices=["openai", "deepseek", "anthropic", "claude"]
    )

    args = parser.parse_args()

    import os

    args.output_file = os.path.join(args.output_folder, args.output_file)
    # args.loc_file = os.path.join(args.output_folder, args.loc_file)

    assert (
            not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    assert (not "deepseek" in args.model) or (
            args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    localize(args)


if __name__ == "__main__":
    main()
