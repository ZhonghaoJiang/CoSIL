import json


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def top_k_accuracy(gt, preds, k):
    return any(item in gt for item in preds[:k])


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
                    predicted_methods.append(entry.split(': ')[1])
    return predicted_methods


def construct_pred_func(file_locs, func_locs):
    final_funcs = []
    # for file_loc in file_locs:
    #     final_funcs.append(func_locs.get(file_loc, []))
    for key, item in func_locs.items():
        final_funcs.append(item)
    return final_funcs


def evaluate_accuracy(loc_outputs, gt_data):
    # Initialize counters for file-level accuracy
    top1_file_correct = 0
    top3_file_correct = 0
    top5_file_correct = 0

    # Initialize counters for function-level accuracy
    top1_func_correct = 0
    top3_func_correct = 0
    top5_func_correct = 0

    # Total instances
    total_instances = len(loc_outputs)

    # Evaluate each instance
    for loc_output in loc_outputs:
        instance_id = loc_output['instance_id']
        print(instance_id)
        # print(loc_output.keys())
        predicted_files = loc_output['found_files']
        pred_funcs = construct_pred_func(predicted_files, loc_output.get('found_related_locs', {}))
        predicted_methods = extract_predicted_methods(pred_funcs)
        # predicted_files = loc_output['file_locations']
        # predicted_methods = extract_predicted_methods(loc_output['func_locations'])

        # pridicted_line = loc_output['found_edit_locs']


        print(f"predicted_files:{predicted_files}, predicted_methods:{predicted_methods}")

        # Ground truth data
        if instance_id in gt_data:
            gt_files, gt_methods = parse_gt_methods(gt_data[instance_id])
            print(f"gt_files:{gt_files}, gt_methods:{gt_methods}")

            # File-level accuracy
            if top_k_accuracy(gt_files, predicted_files, 1):
                top1_file_correct += 1
            if top_k_accuracy(gt_files, predicted_files, 3):
                top3_file_correct += 1
            if top_k_accuracy(gt_files, predicted_files, 5):
                top5_file_correct += 1

            # Function-level accuracy (including classes and methods)
            if top_k_accuracy(gt_methods, predicted_methods, 1):
                top1_func_correct += 1
            if top_k_accuracy(gt_methods, predicted_methods, 3):
                top3_func_correct += 1
            if top_k_accuracy(gt_methods, predicted_methods, 5):
                top5_func_correct += 1

            print(top1_file_correct, top3_file_correct, top5_file_correct)
            print(top1_func_correct, top3_func_correct, top5_func_correct)


    total_instances = 300
    # Calculate accuracy percentages
    top1_file_accuracy = top1_file_correct / total_instances * 100
    top3_file_accuracy = top3_file_correct / total_instances * 100
    top5_file_accuracy = top5_file_correct / total_instances * 100

    top1_func_accuracy = top1_func_correct / total_instances * 100
    top3_func_accuracy = top3_func_correct / total_instances * 100
    top5_func_accuracy = top5_func_correct / total_instances * 100

    return {
        "file_level": {
            "TOP 1": top1_file_accuracy,
            "TOP 3": top3_file_accuracy,
            "TOP 5": top5_file_accuracy
        },
        "function_level": {
            "TOP 1": top1_func_accuracy,
            "TOP 3": top3_func_accuracy,
            "TOP 5": top5_func_accuracy
        }
    }


if __name__ == "__main__":
    # Load data
    loc_outputs = load_jsonl('loc_outputs.jsonl')

    gt_data = load_json('gt.json')
    print(len(loc_outputs))

    # Evaluate accuracy
    accuracy_results = evaluate_accuracy(loc_outputs, gt_data)

    # Print results
    print("File-level accuracy:")
    for k, v in accuracy_results['file_level'].items():
        print(f"{k}: {v:.2f}%")

    print("\nFunction-level accuracy:")
    for k, v in accuracy_results['function_level'].items():
        print(f"{k}: {v:.2f}%")

# 81 82 85 78
# 75 76 77 77