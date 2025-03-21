import json
import os  # 导入 os 模块，用于检查文件是否存在
from datasets import load_from_disk
from get_repo_structure.get_repo_structure import get_project_structure_from_scratch

# 加载数据集
print("加载数据")
swe_bench_data = load_from_disk("./datasets/SWE-bench_Lite_test")

# 逐个处理数据集中的 bug 实例
for bug in swe_bench_data:
    instance_id = bug['instance_id']
    # 构造 JSON 文件路径
    json_file_path = f"./repo_structures_verified/{instance_id}.json"

    # 检查文件是否已经存在，如果存在则跳过
    if os.path.exists(json_file_path):
        print(f"文件 {json_file_path} 已存在，跳过该实例。")
        continue

    # 如果文件不存在，则生成项目结构并保存
    print(f"处理实例 {instance_id}...")
    d = get_project_structure_from_scratch(
        bug["repo"], bug["base_commit"], instance_id, "playground"
    )

    # 将项目结构保存到 JSON 文件中
    with open(json_file_path, "w") as json_file:
        json.dump(d, json_file, indent=4, ensure_ascii=False)

    print(f"已保存 {json_file_path}")
