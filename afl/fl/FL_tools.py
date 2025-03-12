from afl.util.preprocess_data import extract_structure
from afl.util.utils import load_json


def get_functions_of_class(class_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)

    functions_in_class = []
    for item in classes:
        if item['name'] == class_name:
            for method in item['methods']:
                functions_in_class.append(method['name'])

    return str(functions_in_class)


def get_functions_of_file(file_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)
    func_in_file = []
    for item in functions:
        if item['file'] == file_name:
            func_in_file.append(item['name'])

    return str(func_in_file)


def get_classes_of_file(file_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)
    classes_in_file = []
    for item in classes:
        if item['file'] == file_name:
            classes_in_file.append(item['name'])

    return str(classes_in_file)


def get_code_of_file(file_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)
    file_content = "You provide a wrong file name. Please try another file name again."
    for item in files:
        if item[0] == file_name:
            file_content = "\n".join(item[-1])

    return file_content


def get_code_of_class(file_name: str, class_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)

    for item in classes:
        if item['file'] == file_name and item['name'] == class_name:
            return "\n".join(item['class_content'])
    return "You provide a wrong file name or class name. Please try another file name again."


def get_code_of_class_function(file_name: str, class_name: str, func_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)

    for item in classes:
        if item['file'] == file_name and item['name'] == class_name:
            for method in item['methods']:
                if method['name'] == func_name:
                    return "\n".join(method['method_content'])

    return "You provide a wrong file name or class name or function name. Please try another file name again. It may be a file function."


def get_code_of_file_function(file_name: str, func_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)

    for item in functions:
        if item['file'] == file_name and item['name'] == func_name:
            return "\n".join(item['text'])

    return "You provide a wrong file name or function name. Please try another file name again. It may be a class function."

def get_all_of_files(instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)
    file_names = set()
    for item in files:
        file_names.add(item[0])

    return list(file_names)

def get_imports_of_file(file_name: str, instance_id: str):
    d = load_json(f"./repo_structures/{instance_id}.json")
    structure = d["structure"]
    files, classes, functions = extract_structure(structure)
    imports = []
    for item in files:
        if item[0] == file_name:
            for line in item[-1]:
                if line.startswith("import") or (line.startswith("from") and "import" in line):
                    imports.append(line)
    return imports


if __name__ == '__main__':
    # print(get_functions_of_class('WCS', 'astropy__astropy-7746'))
    # print(get_code_of_file_function('astropy/wcs/wcs.py', '_return_single_array', 'astropy__astropy-7746'))
    # print(get_all_of_files('get_all_of_files'))
    print(get_imports_of_file('sympy/matrices/expressions/blockmatrix.py', 'sympy__sympy-17630'))

