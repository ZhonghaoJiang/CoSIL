import json
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any


class DiffNode(BaseModel, frozen=True):
    node_name: str
    node_type: str
    lineno: int
    end_lineno: int

    def __repr__(self):
        return f"{self.node_type}:{self.node_name} {self.lineno}:{self.end_lineno}"


class SrcRange(BaseModel, frozen=True):
    lineno: int
    end_lineno: int
    is_pure_addition: bool
    is_global_addition: bool


class DiffLoc(BaseModel, frozen=True):
    file: str
    diff_nodes: List[DiffNode]
    lineno: int
    end_lineno: int

    def __repr__(self):
        return f"{self.file} {self.lineno}:{self.end_lineno}\n" + "\n".join(
            [
                f"    Node Level {i}: {repr(diff_node)}"
                for i, diff_node in enumerate(self.diff_nodes)
            ]
        )


class ParsedPatch(BaseModel):
    diff_locs: List[DiffLoc]

    def __repr__(self):
        return "\n".join(
            [
                f"Diff Loc {i} : {repr(diff_loc)}"
                for i, diff_loc in enumerate(self.diff_locs)
            ]
        )



def patch_to_location_info(patch_content: str, instance_id: str = "") -> Dict[str, Any]:
    """
    将单个 patch 的内容转换为位置信息，包括文件列表和函数/方法信息。

    :param patch_content: patch 内容的 JSON 字符串
    :param instance_id: 可选参数，实例 ID（用于调试信息）
    :return: 包含文件路径列表和函数信息的字典
    """
    try:
        parsed_patch = ParsedPatch.model_validate_json(patch_content)
    except Exception as e:
        raise ValueError(f"实例 {instance_id} 解析 parsed_patch 失败: {e}")

    file_set = set()
    func_set = set()
    for diff_loc in parsed_patch.diff_locs:
        file_path = diff_loc.file
        file_set.add(file_path)
        diff_nodes = diff_loc.diff_nodes
        if not diff_nodes:
            continue

        # 构造函数名称，格式：文件路径:函数名
        func_name = file_path + "::" + diff_nodes[0].node_name

        # 如果包含 ClassDef 和 FunctionDef，则拼接类名和方法名
        if (
            len(diff_nodes) > 1
            and diff_nodes[0].node_type == "ClassDef"
            and diff_nodes[1].node_type == "FunctionDef"
        ):
            func_name += "." + diff_nodes[1].node_name
        # 若第一个节点不是 FunctionDef（且不是 ClassDef+FunctionDef），记录异常情况并跳过
        elif len(diff_nodes) > 1 and diff_nodes[0].node_type != "FunctionDef":
            print(f"实例 {instance_id} 存在异常 diff_loc: {diff_nodes}")
            continue
        func_set.add(func_name)
    if len(func_set) == 0:
        func_set = file_set
    return {
        "found_files": list(file_set),
        "found_related_locs": list(func_set)
    }


def extract_location_info(ds_golden: pd.DataFrame) -> List[dict]:
    """
    从每个 golden patch 中提取位置信息（文件与函数）。
    调用 patch_to_location_info 完成从 patch 内容到位置信息的转换。
    """
    results = []
    for _, row in ds_golden.iterrows():
        instance_id = row["instance_id"]
        try:
            location_info = patch_to_location_info(row["parsed_patch"], instance_id)
        except Exception as e:
            print(e)
            continue

        results.append({
            "instance_id": instance_id,
            **location_info
        })
    return results


def main():
    # 假设 golden 数据集已保存为 CSV 文件
    golden_csv_path = "./assets/verified_golden_stats.csv"  # 根据实际情况修改路径
    if not os.path.isfile(golden_csv_path):
        print(f"找不到文件 {golden_csv_path}")
        return

    ds_golden = pd.read_csv(golden_csv_path)
    locations = extract_location_info(ds_golden)
    res_dict = {}
    for loc in locations:
        res_dict[loc["instance_id"]] = loc["found_related_locs"]
    print(json.dumps(res_dict, indent=4))
    # 写入文件
    with open("gt_verified.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(res_dict, ensure_ascii=False, indent=4))

    # output_file = "gt_verified.jsonl"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for loc in locations:
    #         f.write(json.dumps(loc, ensure_ascii=False) + "\n")
    # print(f"位置信息已保存到 {output_file}")


if __name__ == "__main__":
    main()
