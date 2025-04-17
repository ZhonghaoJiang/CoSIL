import json


def extract_locations_from_entities(entities):
    found_related_locs = {}
    for entity in entities:
        file_path, func_name = entity.split(':', 1)
        if file_path not in found_related_locs:
            found_related_locs[file_path] = []
        found_related_locs[file_path].append(f"function: {func_name}")
    for k, v in found_related_locs.items():
        found_related_locs[k] = ["\n".join(v)]
    return found_related_locs


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # 加载数据
    loc_outputs_jsonl = load_jsonl('../results/locagent_verified_ori/locagent_qwen_coder_32b_func.jsonl')
    new_outputs = []
    for loc in loc_outputs_jsonl:
        # 从found_entities中提取位置信息
        found_related_locs = extract_locations_from_entities(loc["found_entities"])

        new_outputs.append({
            "instance_id": loc["instance_id"],
            "found_files": loc["found_files"],
            "found_related_locs": found_related_locs
        })

    save_jsonl(new_outputs, '../loc_to_patch_verified/locagent/locagent_qwen_coder_32b_func.jsonl')



