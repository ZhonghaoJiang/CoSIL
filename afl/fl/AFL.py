import re
from abc import ABC
from typing import Any

from FL_prompt import *
from FL_tools import *
from afl.util.api_requests import num_tokens_from_messages
from afl.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from afl.util.preprocess_data import (get_repo_files, get_full_file_paths_and_classes_and_functions, correct_file_paths,
                                      line_wrap_content, transfer_arb_locs_to_locs, show_project_structure,
                                      )



class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement


class AFL(FL):
    def __init__(
            self,
            instance_id,
            structure,
            problem_statement,
            model_name,
            backend,
            logger,
            **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 1024
        self.model_name = model_name
        self.backend = backend
        self.logger = logger

        self.MAX_CONTEXT_LENGTH = 32768 if "qwen2.5-7b" in model_name else 100000

    def _parse_top5_file(self, content: str) -> list[str]:
        # 提取 ``` 包裹的部分
        extracted_output = re.search(r'```(?:.*?)\n(.*?)```', content, re.DOTALL).group(1)
        # 按行分割输出
        lines = extracted_output.strip().split('\n')
        # 提取每行的 PathName.ClassName.MethodName
        parsed_list = []
        for line in lines:
            # 按冒号分割并提取右侧部分
            parsed_list.append(line)
        return parsed_list

    def _parse_output(self, content: str):
        extracted_output = re.search(r'~~~(?:.*?)\n(.*?)~~~', content, re.DOTALL).group(1)
        return extracted_output

    def _issue_clarify(self):
        from afl.util.model import make_model
        clarify_msg = bug_report_clarify_prompt.format(problem_statement=self.problem_statement)
        message = [
            {
                "role": "user",
                "content": clarify_msg
            }
        ]
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=4096,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        bug_report_clarified = self._parse_output(traj["response"])
        return bug_report_clarified

    def consturct_bug_file_list(self, file: list):
        bug_file_content = ""
        for name in file:
            class_content = get_classes_of_file(name, self.instance_id)
            class_list = eval(get_classes_of_file(name, self.instance_id))
            class_func_content = "[\n"
            for class_name in class_list:
                class_func = get_functions_of_class(class_name, self.instance_id)
                class_func_content += f"{class_name}: {class_func} \n"
            class_func_content += "]"
            file_func_content = get_functions_of_file(name, self.instance_id)
            single_file_context = f"file: {name} \n\t class: {class_content} \n\t static functions:  {file_func_content} \n\t class fucntions: {class_func_content}\n"
            bug_file_content += single_file_context
            # print(bug_file_content)
        return bug_file_content

    def localize(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        # lazy import, not sure if this is actually better?

        from afl.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt.format(functions=location_tool_prompt, max_try=max_try)
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt.format(bug_file_list=bug_file_content,
                                                               pre_select_num=7,
                                                               top_n=5)
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        reason = traj["response"]
        current_tokens = traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]

        message.append({
            "role": "assistant",
            "content": reason
        })
        message.append({
            "role": "user",
            "content": call_function_prompt + bug_file_content
        })

        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        }], self.model_name) + self.max_tokens

        for j in range(max_try):

            if current_tokens > self.MAX_CONTEXT_LENGTH - 5 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break

            content = traj["response"]
            print(content)
            message.append({
                "role": "assistant",
                "content": content
            })
            try:
                function_call = content.replace("'", "").replace('"', '').replace("`", "")
                function_name = function_call[:function_call.find('(')].strip()
                arguments = function_call[function_call.find('(') + 1:function_call.rfind(')')].strip()
                args = [arg.strip() for arg in re.split(r",\s*(?![^()]*\))", arguments)]

                # 确保参数数量不超过三个
                arg1, arg2, arg3 = (args + ['None'] * 3)[:3]
                if function_name == 'get_functions_of_class':
                    function_retval = get_functions_of_class(arg1, self.instance_id)
                elif function_name == 'get_code_of_class':
                    function_retval = get_code_of_class(arg1, arg2, self.instance_id)
                elif function_name == 'get_code_of_class_function':
                    function_retval = get_code_of_class_function(arg1, arg2, arg3, self.instance_id)
                elif function_name == 'get_code_of_file_function':
                    function_retval = get_code_of_file_function(arg1, arg2, self.instance_id)
                else:
                    break
                # print(function_retval)
                message.append({"role": "user", "content": function_retval})
                message.append({
                    "role": "user",
                    "content": call_function_prompt + bug_file_content
                })
            except Exception as e:
                print(e)
                message.append({
                    "role": "user",
                    "content": "Please call functions in the right format to get enough information for your final answer." + location_tool_prompt})


        message.append({
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
        )

    def localize_line(
            self,
            file_names,
            func_locs,
            context_window: int = 10,
            add_space: bool = False,
            sticky_scroll: bool = False,
            no_line_number: bool = False,
            temperature: float = 0.0,
            num_samples: int = 1,
    ):
        from afl.util.api_requests import num_tokens_from_messages
        from afl.util.model import make_model
        self.max_tokens = 4096
        coarse_locs = func_locs
        # file_names = []
        if not isinstance(func_locs, dict):
            return [], {}, {}
        # for key, item in func_locs.items():
        #     file_names.append(key)

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        assert num_tokens_from_messages(message, self.model_name) < self.MAX_CONTEXT_LENGTH

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )

    def file_localize_without_collect(
            self, max_retry=10, mock=False
    ) -> tuple[list[str], Any, Any]:
        # lazy import, not sure if this is actually better?

        from afl.util.model import make_model
        max_try = max_retry
        all_files = get_all_of_files(self.instance_id)
        # clarified_issue = self._issue_clarify()

        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=all_files)
        # bug_report = bug_report_template.format(problem_statement=clarified_issue,
        #                                         structure=all_files)

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_try * 0.75),
                                                               top_n=int(max_try / 2))
        user_msg = f"""
                {bug_report}
                {guidence_msg}
                """
        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        reason = raw_output
        message.append({
            "role": "assistant",
            "content": reason
        })
        message.append({
            "role": "user",
            "content": file_summary
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)
        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )
        found_files = correct_file_paths(model_found_files, files)

        return (
            found_files,
            raw_output,
            traj,
        )

    def file_localize(self, max_retry=10, mock=False):
        from afl.util.api_requests import num_tokens_from_messages
        from afl.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=all_files.strip())

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_retry * 0.75),
                                                               top_n=int(max_retry / 2))
        user_msg = f"""
                        {bug_report}
                        {guidence_msg}
                        {file_summary}
                        """

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = self._parse_top5_file(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files)

        self.logger.info(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

def construct_topn_file_context(
        file_to_locs,
        pred_files,
        file_contents,
        structure,
        context_window: int,
        loc_interval: bool = True,
        fine_grain_loc_only: bool = False,
        add_space: bool = False,
        sticky_scroll: bool = False,
        no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals

# if __name__ == '__main__':
#
#
#     print("加载数据")
#     swe_bench_data = load_from_disk("../../datasets/SWE-bench_Lite_test")
#     # bug = swe_bench_data[5]
#     bug = [x for x in swe_bench_data if x["instance_id"] == "sympy__sympy-13471"][0]
#     problem_statement = bug["problem_statement"]
#     instance_id = bug["instance_id"]
#     print(problem_statement)
#     print(instance_id)
#     d = load_json(f"../../repo_structures/{instance_id}.json")
#     structure = d["structure"]
#     import logging
#     fl = AFL(
#         d["instance_id"],
#         structure,
#         problem_statement,
#         "deepseek-coder",
#         "deepseek",
#         logging.getLogger("AFL"),
#     )
#     print("------------------------------")
#     print("start localization")
#     ret = fl.file_localize()[0]
#     print(ret)
#     loc = fl.localize(file=ret)[0]
#     print(loc)
#     line_loc = fl.localize_line(ret, loc, temperature=0.85, num_samples=1)[0]
#     print(line_loc)
