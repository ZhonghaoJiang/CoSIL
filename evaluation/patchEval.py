import json
import re
from tqdm import tqdm
from afl.util.utils import load_json
from afl.util.preprocess_data import extract_structure

REPO_STRUCTURE = None

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def extract_file(diff_text):
    file_pattern = r"^diff --git a\/(.+?) b\/\1"
    files = re.findall(file_pattern, diff_text, re.MULTILINE)
    return files

def extract_line(diff_text):
    lines_pattern = r"@@ -(\d+),(\d+) \+(\d+),(\d+)"
    line_changes = re.findall(lines_pattern, diff_text)
    lines = [int(int(l[0]) + int(l[1])/2) for l in line_changes]
    return lines

def get_function_from_line(file_name: str, line: int, instance_id: str):
    line = int(line)
    d = REPO_STRUCTURE[instance_id]
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)

    for item in functions:
        if item['file'] == file_name and int(item['start_line']) <= line <= int(item['end_line']):
            return item['name']

    for item in classes:
        if item['file'] == file_name and int(item['start_line']) <= line <= int(item['end_line']):
            for method in item['methods']:
                if int(method['start_line']) <= line <= int(method['end_line']):
                    return f"{item['name']}.{method['name']}"
            return item['name']

    return None

def eval_acc(patches, gt):
    acc = 0

    for patch in tqdm(patches, desc="Evaluating patches"):
        if patch['model_patch']=="":
            continue
        instance_id = patch['instance_id']
        file_name = extract_file(patch['model_patch'])[0]
        line_list = extract_line(patch['model_patch'])
        function_list = []
        for line in line_list:
            function = get_function_from_line(file_name, line, instance_id)
            function_list.append(f"{file_name}::{function}")
            if set(function_list).issubset(set(gt[instance_id])):
                acc += 1

    return acc / 300

if __name__ == '__main__':
    # patches = load_jsonl('../test_patches/opencsg.jsonl')
    gt_data = load_json('gt.json')
    REPO_STRUCTURE = {instance_id: load_json(f"../repo_structures/{instance_id}.json") for instance_id in
                      tqdm(gt_data.keys())}
    path_list = [
        "../test_patches/agentless_1.5.jsonl",
        "../test_patches/autocoderover.jsonl",
        "../test_patches/sweagent.jsonl",
        "../all_preds_afl_gpt.jsonl",
        "../all_preds_agentless_gpt.jsonl",
        "../all_preds_orcaloca_gpt.jsonl"
    ]
    for p in path_list:
        patches = load_jsonl(p)
        acc = eval_acc(patches, gt_data)
        print(p, acc)

