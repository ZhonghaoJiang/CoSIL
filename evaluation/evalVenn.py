import argparse
import json

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_gt_methods(gt_entries):
    """
    解析ground truth中的条目，并统一处理为文件级别和函数级别的定位。
    """
    files, methods = set(), set()

    for entry in gt_entries:
        parts = entry.split('::')

        if len(parts) == 2:  # File::Method 或 File::Class
            file_name, method_or_class = parts
            files.add(file_name)
            methods.add(method_or_class)

        elif len(parts) == 1:  # File
            files.add(parts[0])

    return files, methods


def extract_predicted_methods(found_related_locs):
    """
    从 found_related_locs 中提取预测的类和函数名称。
    """
    predicted_methods = []
    for sublist in found_related_locs:
        for loc in sublist:
            for entry in loc.split('\n'):
                if 'function:' in entry or 'class:' in entry:
                    try:
                        predicted_methods.append(entry.split(': ')[1])
                    except:
                        pass
    return predicted_methods


def construct_pred_func(file_locs, func_locs):
    final_funcs = []
    # for file_loc in file_locs:
    #     final_funcs.append(func_locs.get(file_loc, []))
    for key, item in func_locs.items():
        final_funcs.append(item)
    return final_funcs

def top_k_accuracy(gt, preds, k):
    return any(item in gt for item in preds[:k])

def extract_top5_intance_ids(afl, agentless, orcaloca, gt):
    to_eval = [afl, agentless, orcaloca]
    afl_instance_file, agentless_instance_file, orcaloca_instance_file = [], [], []
    afl_instance_func, agentless_instance_func, orcaloca_instance_func = [], [], []
    to_store_file = [afl_instance_file, agentless_instance_file, orcaloca_instance_file]
    to_store_func = [afl_instance_func, agentless_instance_func, orcaloca_instance_func]
    for loc_outputs, res_file, res_func in zip(to_eval, to_store_file, to_store_func):
        for loc_output in loc_outputs:
            instance_id = loc_output['instance_id']
            # print(instance_id)
            predicted_files = loc_output['found_files']
            if not predicted_files:
                continue
            pred_funcs = construct_pred_func(predicted_files, loc_output.get('found_related_locs', {}))
            predicted_methods = extract_predicted_methods(pred_funcs)
            if instance_id in gt_data:
                gt_files, gt_methods = parse_gt_methods(gt_data[instance_id])
                # print(f"gt_files:{gt_files}, gt_methods:{gt_methods}")

                # 计算TOPN准确率（文件级）
                if top_k_accuracy(gt_files, predicted_files, 5):
                    res_file.append(instance_id)

                if top_k_accuracy(gt_methods, predicted_methods, 5):
                    res_func.append(instance_id)

    return to_store_file, to_store_func

def extract_afl_special(to_store_file, to_store_func):
    a, b, c = to_store_file
    x, y, z = to_store_func
    return set(a) - (set(b) | set(c)), set(x) - (set(y) | set(z))

if __name__ == '__main__':
    afl = load_jsonl('../loc_to_patch/afl/loc_qwen_coder_32b_func.jsonl')
    agentless = load_jsonl('../loc_to_patch/agentless/agentless_qwen_coder_32b_func.jsonl')
    orcaloca = load_jsonl('../loc_to_patch/orcaloca/orca_qwen_coder_32b_func.jsonl')

    gt_data = load_json('gt.json')
    # gt_data = load_json('gt_verified.json')

    to_store_file, to_store_func = extract_top5_intance_ids(afl, agentless, orcaloca, gt_data)

    u_file, u_func = extract_afl_special(to_store_file, to_store_func)

    print(u_file)
    print(u_func)
    print(u_file & u_func)

"""
AFL
django__django-13315
predicted_files:['django/db/models/fields/related.py', 'django/db/models/query.py', 'django/db/models/sql/compiler.py', 'django/db/models/sql/where.py', 'django/forms/models.py'], predicted_methods:['apply_limit_choices_to_to_formfield', 'ModelChoiceField._get_choices', 'ForeignKey.get_limit_choices_to', 'QuerySet.filter', 'SQLCompiler.get_distinct']
gt_files:{'django/forms/models.py'}, gt_methods:{'apply_limit_choices_to_to_formfield'}

Agentless
predicted_files:['django/forms/fields.py', 'django/db/models/fields/related.py', 'django/db/models/fields/related_descriptors.py', 'django/db/models/query.py', 'django/db/models/sql/compiler.py'], predicted_methods:['ForeignKey.formfield', 'ForeignKey.get_limit_choices_to', 'ForeignObject', 'ForeignObject.get_limit_choices_to', 'RelatedField', 'RelatedField.get_limit_choices_to', 'QuerySet', 'QuerySet.filter', 'QuerySet.distinct', 'SQLCompiler', 'SQLCompiler.get_related_selections', 'SQLCompiler.get_distinct']
gt_files:{'django/forms/models.py'}, gt_methods:{'apply_limit_choices_to_to_formfield'}

Orcaloca
predicted_files:['django/db/models/fields/related.py'], predicted_methods:['ForeignKey.formfield', 'ForeignKey']
gt_files:{'django/forms/models.py'}, gt_methods:{'apply_limit_choices_to_to_formfield'}.


ForeignKey.formfield() <- RelatedField.formfield() <- Field.formfield() <- ModelFormMetaclass.__new__() -> apply_limit_choices_to_to_formfield()

"""