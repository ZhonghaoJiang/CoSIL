import random
from datasets import load_from_disk, Dataset

# 加载数据集
print("加载数据")
swe_bench_data = load_from_disk("./datasets/SWE-bench_Verified_sampled")

# 随机采样50条数据
print("采样数据")
sampled_data = swe_bench_data.shuffle(seed=42).select(range(50))

# 保存采样后的数据
output_dir = "./datasets/SWE-bench_Verified_sampled"
sampled_data.save_to_disk(output_dir)

print(f"采样后的数据已保存到 {output_dir}")

