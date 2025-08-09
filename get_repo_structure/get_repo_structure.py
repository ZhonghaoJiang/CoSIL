import argparse
import ast
import json
import os
import subprocess
import uuid

import pandas as pd
from tqdm import tqdm

MIRROR_ROOT = "mirror_cache"          # 本地镜像存放目录

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# def clone_repo(repo_name, repo_playground):
#     try:
#
#         print(
#             f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_name]}..."
#         )
#         subprocess.run(
#             [
#                 "git",
#                 "clone",
#                 f"https://github.com/{repo_name}.git",
#                 f"{repo_playground}/{repo_to_top_folder[repo_name]}",
#             ],
#             check=True,
#         )
#         print("Repository cloned successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while running git command: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

def ensure_local_mirror(repo_name: str, mirror_root: str = MIRROR_ROOT):
    """
    确保本地存在 repo 的裸镜像，若无则 --mirror 克隆，若已有可按需 update。
    返回镜像路径。
    """
    top_folder = repo_to_top_folder[repo_name]
    mirror_path = os.path.join(mirror_root, f"{top_folder}.git")  # 裸仓库

    if not os.path.exists(mirror_path):
        os.makedirs(mirror_root, exist_ok=True)
        print(f"[mirror] cloning {repo_name} ...")
        subprocess.run(
            ["git", "clone", "--mirror", f"https://gh.xmly.dev/https://github.com/{repo_name}.git", mirror_path],
            check=True,
        )
    else:
        # 如希望每次都同步远端可保留下面两行；若网络慢可注释掉
        print(f"[mirror] updating {repo_name} ...")
        # subprocess.run(["git", "-C", mirror_path, "remote", "update", "--prune"], check=True)

    return mirror_path


def clone_repo(repo_name, repo_playground, mirror_root: str = MIRROR_ROOT):
    """
    先确认本地镜像；随后用镜像快速派生工作副本。
    """
    top_folder = repo_to_top_folder[repo_name]
    dest_path = os.path.join(repo_playground, top_folder)

    # 已存在工作副本则直接返回
    if os.path.exists(dest_path):
        print(f"[skip] {dest_path} already exists")
        return

    # 1) 确保镜像存在
    mirror_path = ensure_local_mirror(repo_name, mirror_root)

    # 2) 用本地镜像克隆工作副本（--shared 硬链接，速度快且节省空间）
    print(f"[clone] from local mirror {mirror_path} -> {dest_path}")
    subprocess.run(
        ["git", "clone", "--shared", mirror_path, dest_path],
        check=True,
    )
    print("[clone] repository cloned successfully from mirror")



def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_name]}", commit_id)
    structure = create_structure(f"{repo_playground}/{repo_to_top_folder[repo_name]}")
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_name]}"], check=True
    )
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d

def get_dependency_of_repo(
    repo_name, commit_id, instance_id, repo_playground, folder
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_name]}", commit_id)
    parse_dependency(f"{repo_playground}/{repo_to_top_folder[repo_name]}", instance_id, folder)
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_name]}"], check=True
    )

def parse_dependency(directory_path, instance_id, folder):
    subprocess.run(
        [
            'java', '-Xmx8g', '-jar', 'depends.jar',
            '-f', 'json',
            '--granularity', 'method',
            '--detail',
            '-s',
            'python', directory_path, f'{folder}/{instance_id}'
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL  # 或者 subprocess.STDOUT
    )



def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure
