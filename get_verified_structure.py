import json
import os  # 导入 os 模块，用于检查文件是否存在
from datasets import load_from_disk
from get_repo_structure.get_repo_structure import get_project_structure_from_scratch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 定义处理单个实例的函数
def process_bug(bug):
    instance_id = bug['instance_id']
    # 构造 JSON 文件路径
    json_file_path = f"./repo_structures/{instance_id}.json"

    # 检查文件是否已经存在，如果存在则跳过
    if os.path.exists(json_file_path):
        return f"文件 {json_file_path} 已存在，跳过该实例。"

    # 如果文件不存在，则生成项目结构并保存
    d = get_project_structure_from_scratch(
        bug["repo"], bug["base_commit"], instance_id, "playground"
    )

    # 将项目结构保存到 JSON 文件中
    with open(json_file_path, "w") as json_file:
        json.dump(d, json_file, indent=4, ensure_ascii=False)

    return f"已保存 {json_file_path}"

# 加载数据集
print("加载数据")
swe_bench_data = load_from_disk("./datasets/SWE-bench_Verified_test")

# 使用多进程池处理数据集中的实例，并显示进度条
if __name__ == '__main__':
    # 创建一个进程池，并设置最大进程数为 CPU 核心数
    num_workers = min(os.cpu_count(), 32)  # 可根据需要调整最大进程数

    # 使用 ProcessPoolExecutor 创建多进程池
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用字典保存 Future 对象和对应的任务
        future_to_bug = {executor.submit(process_bug, bug): bug for bug in swe_bench_data}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_bug), total=len(swe_bench_data)):
            # 获取执行结果（如果需要查看返回信息，可以使用 `future.result()`）
            try:
                future.result()  # 这样可以捕获潜在的异常，并且不会抛出到主进程
            except Exception as exc:
                instance_id = future_to_bug[future]['instance_id']
                print(f"实例 {instance_id} 处理时发生异常：{exc}")

    print("所有实例处理完毕。")
