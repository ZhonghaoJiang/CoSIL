import json


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
    loc_outputs_json = load_json('./assets/orcar_verified_parsed_output_output_14b.json')
    instance_ids = loc_outputs_json.keys()
    loc_outputs = [
        {
            "instance_id": instance_id,
            "found_files": loc_outputs_json[instance_id].get("file", {}).get("model", None),
            "found_related_locs": loc_outputs_json[instance_id].get("func", {}).get("model", None)
        }
        for instance_id in instance_ids
    ]

    gt_data = load_json('gt.json')
    print(len(loc_outputs))
    print(loc_outputs)

    for loc in loc_outputs:
        func_list = loc['found_related_locs']
        found_files = loc['found_files']
        found_related_locs = {}
        if not found_files:
            loc['found_files'] = []
            loc['found_related_locs'] = found_related_locs
            continue
        for k in found_files:
            found_related_locs[k] = []
        if len(func_list) == 0:
            loc['found_related_locs'] = found_related_locs
        for func in func_list:
            try:
                a, b = func.split(':')
            except:
                b = func.split(':')[-1]
                a = func.split(':')[-2]
            if a in found_related_locs:
                if b[0].isupper():
                    found_related_locs[a].append("class: " + b)
                else:
                    found_related_locs[a].append("function: " + b)

        for k, v in found_related_locs.items():
            found_related_locs[k] = ["\n".join(v)]

        loc['found_related_locs'] = found_related_locs

    print(loc_outputs)
    save_jsonl(loc_outputs, '../loc_to_patch_verified/orcaloca/orcaloca_qwen_coder_14b_func.jsonl')



