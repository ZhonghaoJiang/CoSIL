# ---------------------------------------------FlexFL-------------------------------------------------------------------

function_prompt = """
Function calls you can use are as follows:
* get_all_of_files() -> Get all the file names of a python project. *
* get_functions_of_class('class_name') -> Get the functions of a specified class in the given python project. 'class_name' -> The name of the class. *
* get_functions_of_file('file_name') -> Get the functions of a specified file in the given python project. 'file_name' -> The name of the file. *
* get_classes_of_file('file_name') -> Get the classes of a specified file except class functions in the given python project. 'file_name' -> The name of the file. *
* get_code_of_file('file_name') -> Get the code of a specified file in the given python project. 'file_name' -> The name of the file. *
* get_code_of_class('file_name', 'class_name') -> Get the code of a specified class in the given file and python project. 'file_name' -> The name of the file. 'class_name' -> The name of the class. *
* get_code_of_class_function('file_name', 'class_name', 'func_name') -> Get the code of a specified function in the given class, file, and python project. 'file_name' -> The name of the file. 'class_name' -> The name of the class. 'func_name' -> The name of the function. *
* get_code_of_file_function('file_name', 'func_name') -> Get the code of a specified function in the given file and python project. 'file_name' -> The name of the file. 'func_name' -> The name of the function. *
* exit() -> Exit function calling to give your final answer when you are confident of the answer. *
"""

bug_report_template_wo_repo_struct = """
The bug report is as follows:
```
### GitHub Problem Description ###
{problem_statement}

###
"""

summary_template = """
Based on the available information, provide complete name of the top-5 most likely culprit methods for the bug please. 
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
```
Top_1 : PathName::ClassName.MethodName
Top_2 : PathName::ClassName.MethodName
Top_3 : PathName::MethodName
Top_4 : PathName::MethodName
Top_5 : PathName::MethodName
```
"""

system_prompt = """
You are a debugging assistant of our Python software.
You will be presented with a bug report and tools (functions) to access the source code of the system under test (SUT).
Your task is to locate the top-5 most likely culprit methods based on the bug report and the information you retrieve using given functions.
{functions}
You have {max_try} chances to call function.
Since functions may exist independently or be defined in classes in a Python file, I suggest that you check the file contents using 'get_code_of_file' after determining the file name to help you locate the next step.
The values of the parameter 'file_name' are in the return list of 'get_all_of_files'.
The values of the parameter 'class_name' are in the return list of 'get_classes_of_file'.
The values of the parameter 'func_name' are in the return list of 'get_functions_of_file' or 'get_functions_of_class'.
To ensure the correctness of the parameters you pass, please call 'get_all_of_files' first to view the file names that can be passed in.
"""
# ------------------------------------------------Bug Report Clarify----------------------------------------------------
bug_report_clarify_prompt = """
I will give you a description of an issue in a GitHub repository. 
Please clarify the place where the natural language contains ambiguity and output the Bug Report in the following form. 
The content you output needs to be between ~~~.
If there is a related description in the Issue Description, you need to summarize it.
If it does not exist, please answer truthfully without existence.
The specific form is as follows:
~~~
[Issue Content Summary]
xxxxxxxxxx

[Files and Code Mentioned in Issue]
xxxxxxxxxx

[User's Traceback]
xxxxxxxxxx

[Reproduce Method Provided by User]
xxxxxxxxxx

[User's Environment Setup]
xxxxxxxxxx

[Possible Solutions Proposed by User]
xxxxxxxxxx

~~~

Original Issue Description is as follows:
{problem_statement}
"""

# ------------------------------------------------AFL-------------------------------------------------------------------
bug_report_template = """
The bug report is as follows:
```
### GitHub Problem Description ###
{problem_statement}

###

### Candidate Files ###
{structure}

###
```
"""

file_system_prompt = """
You will be presented with a bug report with repository structure and tools (functions) to access the source code of the system under test (SUT).
Your task is to locate the top-5 most likely culprit files based on the bug report and the information you retrieve using given functions.
{functions}
You have {max_try} chances to call function.
You can only call 1 function once.
The formal parameter file_name of function 'get_code_of_file' takes the fullpath from the Candidate Files.
"""

file_system_prompt_without_tool = """
You will be presented with a bug report with repository structure to access the source code of the system under test (SUT).
Your task is to locate the top-5 most likely culprit files based on the bug report.
"""

file_tool_prompt = """
Function calls you can use are as follows:
* get_code_of_file('file_name') -> Get the code of a specified file in the given python project. 'file_name' -> The name of the file. *
* exit() -> Exit function calling to give your final answer when you are confident of the answer. *
"""

file_summary = """
Based on the available information, reconfirm and provide complete name of the top-5 most likely culprit files for the bug. 
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```.
```
file1.py
file2.py
file3.py
file4.py
file5.py
```
Replace the 'file1.py' with the actual file path.
For example, 
```
sklearn/linear_model/__init__.py
sklearn/base.py
```
"""

file_summary_top1 = """
Based on the available information, reconfirm and provide complete name of the top-1 most likely culprit files for the bug. 
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
```
file1.py
```
"""

file_guidence_prmpt = """
Let's locate the faulty file step by step using reasoning. 
In order to locate accurately, you can pre-select {pre_select_num} files, then check them, and finally confirm {top_n} file names.
Avoid making multiple identical calls to save overhead.
"""

file_guidence_prmpt_without_tool = """
Let's locate the faulty file step by step using reasoning. 
In order to locate accurately, you can pre-select {pre_select_num} files, then check them through function calls, and finally confirm {top_n} file names.
"""

location_system_prompt = """
You will be presented with a bug report and tools (functions) to access the source code of the system under test (SUT).
Since the modification is based on the code repository, the modified locations may include files, classes, and functions, and the modifications may be in the form of addition, deletion, or update.
Your task is to locate the top-5 most likely culprit locations based on the bug report and the information you retrieve using given functions.
{functions}
You have {max_try} chances to call function.
"""

location_system_prompt_ablation = """
You will be presented with a bug report to access the source code of the system under test (SUT).
Since the modification is based on the code repository, the modified locations may include files, classes, and functions, and the modifications may be in the form of addition, deletion, or update.
Your task is to locate the top-5 most likely culprit locations based on the bug report.
"""

location_guidence_prmpt = """
Let's locate the faulty file step by step using reasoning and function calls. 
I have pre-identified top-5 files that may contain bugs. There stuctures are as follows:
{bug_file_list}
The formal parameter 'file_name' takes the value in "file:"
The formal parameter 'ckass_name' takes the value in "class:"
The formal parameter 'func_name' takes the value in "static functions:" and "class functions: "
Avoid making multiple identical calls to save overhead.
You must strictly follow the structure I give to call different tools.
For static functions, you can use 'get_code_of_file_function', and for class functions, you can use 'get_code_of_class_function'.
In order to locate accurately, you can pre-select {pre_select_num} locations, then check them through function calls, and finally confirm {top_n} file names.
Don't make the first function call in this message.
"""

location_guidence_prmpt_ablation = """
Let's locate the faulty file step by step by reasoning. 
I have pre-identified top-5 files that may contain bugs.
There stuctures are as follows:
{bug_file_list}
"""

location_tool_prompt = """
Function calls you can use are as follows:
* get_code_of_class('file_name', 'class_name') -> Get the code of a specified class in the given file and python project. 'file_name' -> The name of the file. 'class_name' -> The name of the class. *
* get_code_of_class_function('file_name', 'class_name', 'func_name') -> Get the code of a specified function in the given class, file, and python project. 'file_name' -> The name of the file. 'class_name' -> The name of the class. 'func_name' -> The name of the function. *
* get_code_of_file_function('file_name', 'func_name') -> Get the code of a specified function in the given file and python project. 'file_name' -> The name of the file. 'func_name' -> The name of the function. *
* exit() -> Exit function calling to give your final answer when you are confident of the answer. *
"""

location_summary = """
Based on the available information, reconfirm and provide complete name of the top-5 most likely culprit locations for the bug. 
Before make the final decision, please check wether the function name is correct or not, for static functions, don't add class name.
{bug_file_list}

Please provide the complete set of locations as either a class name, a function name, or a variable name.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
Since your answer will be processed automatically, please give your answer in the exapmle format as follows.
```
top1_file_fullpath.py
function: Class1.Function1

top2_file_fullpath.py
function: Function2

top3_file_fullpath.py
class: Class3

top4_file_fullpath.py
function: Class4.Function4

top5_file_fullpath.py
function: Function5
```
Replace the 'Top_file_fullpath.py' with the actual file path, the 'Class' with the actual class name and the 'Function' with the actual function name.
For example, 
```
sklearn/linear_model/__init__.py
function: LinearRegression.fit
```
"""

location_summary_ablation = """
Based on the available information, reconfirm and provide complete name of the top-5 most likely culprit locations for the bug. 
Please provide the complete set of locations as either a class name, a function name, or a variable name.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
Since your answer will be processed automatically, please give your answer in the format as follows.
```
Top1_file_fullpath.py
class: Class1

Top2_file_fullpath.py
function: Function2

Top3_file_fullpath.py
function: Class3.Function3

Top4_file_fullpath.py
function: Class4.Function4

Top5_file_fullpath.py
function: Function5
```
Replace the 'Top1_file_fullpath.py' with the actual file path and the 'Class1' with the actual class name.
For example, 
```
sklearn/linear_model/__init__.py
class: LinearRegression
```
After make the final decision, please check wether the function name is correct or not, for static functions, don't add class name.
"""

call_function_prompt = """
Now call a function in this format 'FunctionName(Argument)' in a single line without any other word or signal (such as ```).
Don't call the same function you've previous called, because this may waste your context length.
"""

format_correct_prompt = """
Here is a localization result, but it seems not in the correct format. Please correct it.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
This is an example of expected output:
```
sklearn/linear_model/__init__.py
sklearn/base.py
```

Please help me corrct the following result.
{res}
"""

file_reflection_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.
I have already find 5 relevent files. Accrording to the import relations, construct the call graph first.
Then, Rank them again and reflect the result.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

### Files To Be Ranked ###
{pre_files}

###

### Import Relations ###
{import_content}
###


Please only provide the full path and return top 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
file3.py
file4.py
file5.py
```
Note: file1.py indicates the top-1 file, file2.py indicates the top-2 file, and so on. Do not include test files.
"""



file_prompt_new = """
You will be presented with a bug report with repository structure to access the source code of the system under test (SUT).
Your task is to locate the most likely culprit locations based on the bug report and return in the following format.
```
PathName::ClassName.MethodName
PathName::MethodName
```
The returned files should be separated by new lines ordered by most to least important and wrapped with ```.
For example:
```
sympy/geometry/point.py::Point
```
"""

# ------------------------------------------------AFL Patch-------------------------------------------------------------
file_patch_system_prompt_without_tool = """
You will be presented with a bug report with repository structure and to access the source code of the system under test (SUT).
You need to add a test case in the repository's test files to verify the correctness of my modifications based on the patches I have generated. 
To ensure the test case runs properly, you need to locate the correct files.
Based on the location of my modifications and the bug report, provide the top-5 files where the test case is most likely to be added.
"""

file_patch_guidence_prmpt = """
Let's locate the file need to be added test cases step by step using reasoning. 
In order to locate accurately, you can pre-select {pre_select_num} files, then check, and finally confirm {top_n} file names.

The patches given to repair this bug as follows:
{patch_list}
"""

file_patch_summary = """
Based on the available information, reconfirm and provide complete name of the top-5 most likely files for adding tests. 
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
```
file1.py
file2.py
file3.py
file4.py
file5.py
```
"""

location_patch_system_prompt = """
You will be presented with a bug report and tools (functions) to access the source code of the system under test (SUT).
You need to add a test case in the repository's test files to verify the correctness of my modifications based on the patches I have generated. 
To ensure the test case runs properly, you need to locate the correct files.
Based on the location of my modifications and the bug report, provide the top-5 locations where the test case is most likely to be added.
Since the modification is based on the code repository, the modified locations may include files, classes, and functions, and the modifications may be in the form of addition, deletion, or update.
Your task is to locate the top-5 most likely locations based on the bug report and the information you retrieve using given functions.
{functions}
You have {max_try} chances to call function.
"""

location_patch_guidence_prmpt = """
Let's locate the file to add test cases step by step using reasoning and function calls. 
I have pre-identified top-5 files that may contain bugs. There stuctures are as follows:
{bug_file_list}
The formal parameter 'file_name' takes the value in "file:"
The formal parameter 'ckass_name' takes the value in "class:"
The formal parameter 'func_name' takes the value in "static functions:" and "class functions: "
Avoid making multiple identical calls to save overhead.
You must strictly follow the structure I give to call different tools.
For static functions, you can use 'get_code_of_file_function', and for class functions, you can use 'get_code_of_class_function'.
In order to locate accurately, you can pre-select {pre_select_num} locations, then check them through function calls, and finally confirm {top_n} file names.
Don't make the first function call in this message.

The patches given to repair this bug as follows:
{patch_list}
"""

location_patch_summary = """
Based on the available information, reconfirm and provide complete name of the top-5 most likely culprit locations for the bug. 
Since your answer will be processed automatically, please give your answer in the format as follows.
Please provide the complete set of locations as either a class name, a function name, or a variable name.
If the location to be modified is a function in the entire class or file, it returns "class: ClassName" or "function: FunctionName"; if it is a class function, it returns "function: ClassName.FunctionName"
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
```
Top1_file_fullpath/file1.py
class: Class1

Top2_file_fullpath/file2.py
function: Function2

Top3_file_fullpath/file3.py
function: Class3.Function3

Top4_file_fullpath/file4.py
function: Class4.Function4

Top5_file_fullpath/file5.py
function: Function5
```
After make the final decision, please check wether the function name is correct or not, for static functions, don't add class name.
{bug_file_list}
"""
# ------------------------------------------------Agentless-------------------------------------------------------------
file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""

obtain_relevant_functions_and_vars_from_raw_files_prompt = """
Please look through the following GitHub Problem Description and Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

file_content_template = """
### File: {file_name} ###
{file_content}
"""

obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
If the location to be edited is a function in the entire class or file, it returns "class: ClassName" or "function: FunctionName" and line numbers; if it is a class function, it returns "function: ClassName.FunctionName" and line numbers.
### Examples:
```
FullFileName1.py
line: 10
class: Class1
line: 51

FullFileName2.py
function: Class2.Function1
line: 12

FullFileName3.py
function: Function2
line: 24
line: 156
```

The returned content must be wrapped with ```
"""
obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

full_path3/file3.py
function: my_function2
```

Return just the location(s)
"""

obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

obtain_irrelevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of folders that are irrelevant to fixing the problem.
Note that irrelevant folders are those that do not need to be modified and are safe to ignored when trying to solve this problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path.
Remember that any subfolders will be considered as irrelevant if you provide the parent folder.
Please ensure that the provided irrelevant folders do not include any important files needed to fix the problem
The returned folders should be separated by new lines and wrapped with ```
For example:
```
folder1/
folder2/folder3/
folder4/folder5/
```
"""