import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from afl.util.utils import load_json, load_jsonl
from afl.util.preprocess_data import extract_structure


REPO_STRUCTURE = {}
STRUCTURE_INDEX = {}

def process_instance(instance_id):
    try:
        d = load_json(f"../repo_structures/{instance_id}.json")
        structure = d.get("structure", [])
        _, classes, functions = extract_structure(structure)

        func_map = {}
        for f in functions:
            func_map.setdefault(f['file'], []).append((int(f['start_line']), int(f['end_line']), f['name']))

        class_map = {}
        for c in classes:
            entries = []
            for m in c.get('methods', []):
                full_name = f"{c['name']}.{m['name']}"
                entries.append((int(m['start_line']), int(m['end_line']), full_name))
            entries.append((int(c['start_line']), int(c['end_line']), c['name']))
            class_map.setdefault(c['file'], []).extend(entries)

        return instance_id, d, {"func_map": func_map, "class_map": class_map}
    except Exception as e:
        print(f"[ERROR] Failed to process {instance_id}: {e}")
        return instance_id, None, None

from multiprocessing import cpu_count

def load_all_structures_parallel(instance_ids):
    global REPO_STRUCTURE, STRUCTURE_INDEX
    with ProcessPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(process_instance, iid): iid for iid in instance_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Building structure index"):
            instance_id, repo_data, index = future.result()
            if repo_data and index:
                REPO_STRUCTURE[instance_id] = repo_data
                STRUCTURE_INDEX[instance_id] = index


def get_function_from_line(file_name: str, line: int, instance_id: str):
    line = int(line)
    if instance_id not in STRUCTURE_INDEX:
        return None

    func_map = STRUCTURE_INDEX[instance_id]["func_map"]
    class_map = STRUCTURE_INDEX[instance_id]["class_map"]

    for start, end, name in func_map.get(file_name, []):
        if start <= line <= end:
            return name

    for start, end, name in class_map.get(file_name, []):
        if start <= line <= end:
            return name

    return None

def extract_file_to_old_lines(diff_text):
    """
    解析 diff，返回一个 dict: {filename: [old_lines]}，
    表示每个文件中受修改的旧行号（原始行号）列表。
    """
    file_to_lines = {}
    current_file = None

    lines = diff_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git"):
            match = re.match(r"^diff --git a\/.+? b\/(.+)", line)
            if match:
                current_file = match.group(1)
                file_to_lines[current_file] = []
        elif line.startswith("@@") and current_file:
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", line)
            if match:
                old_start = int(match.group(1))
                old_len = int(match.group(2)) if match.group(2) else 1
                # 加入 old 区间中间的那一行，代表修改大致区域
                file_to_lines[current_file].append(old_start + old_len // 2)

    return file_to_lines

def eval_acc(patches, gt):
    acc_f = 0
    acc_func = 0
    total = 300  # 固定样本总数，不根据有效样本动态调整

    for patch in tqdm(patches, desc="Evaluating patches"):
        if not patch.get('model_patch'):
            continue
        instance_id = patch.get('instance_id')
        if not instance_id or instance_id not in gt:
            continue

        try:
            file_line_map = extract_file_to_old_lines(patch['model_patch'])
        except Exception:
            continue

        if not file_line_map:
            continue

        gt_methods = set(gt[instance_id])
        gt_files = set(x.split("::")[0] for x in gt_methods)
        has_function_level_gt = any("::" in x for x in gt_methods)

        matched_funcs = set()
        matched_files = set()

        for file_name, old_lines in file_line_map.items():
            for line in old_lines:
                function = get_function_from_line(file_name, line, instance_id)
                if function:
                    matched_funcs.add(f"{file_name}::{function}")
                matched_files.add(file_name)

        if matched_files & gt_files:
            acc_f += 1

        if not has_function_level_gt:
            acc_func += 1
        elif matched_funcs & gt_methods:
            acc_func += 1

    return acc_f / total, acc_func / total


if __name__ == '__main__':
    gt_data = load_json('gt.json')
    instance_ids = list(gt_data.keys())
    load_all_structures_parallel(instance_ids)

    path_list = [
        "./openhands.jsonl",
    ]
    for p in path_list:
        patches = load_jsonl(p)
        file_match, function_match = eval_acc(patches, gt_data)
        print(p, file_match, function_match)

