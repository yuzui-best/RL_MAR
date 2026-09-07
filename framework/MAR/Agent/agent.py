from typing import Dict
import hashlib
import time
import json
import os
from loguru import logger

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_registry import LLMRegistry
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation_plc import message_aggregation #,inner_test
from MAR.Prompts.post_process_plc import (
    post_process,
    _extract_st_code,
    _format_codesys_errors,
    _build_rename_hint,
    canonicalize_st,
    replace_st_code,
    functional_context,
    functional_available,
    functional_repair_instructions,
    run_functional_check,
    sanitize_compiler_diagnostic,
    public_functional_summary,
    functional_feedback,
    functional_score,
)
from MAR.Prompts.output_format_plc import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt
from MAR.Utils.ablation_config import get_ablation_mode


@AgentRegistry.register('Agent')
class Agent(Node):
    def __init__(self, id: str | None =None, domain: str = "", role:str = None , llm_name: str = "",reason_name: str = "",):
        super().__init__(id, reason_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.role = RoleRegistry(domain, role)
        self.reason = reason_name

        self.message_aggregation = self.role.get_message_aggregation()
        self.description = self.role.get_description()
        self.output_format = self.role.get_output_format()
        self.post_process = self.role.get_post_process()
        self.post_description = self.role.get_post_description()
        self.post_output_format = self.role.get_post_output_format()
        # Reflect
        if reason_name == "Reflection" and self.post_output_format == "None":
            self.post_output_format = self.output_format
            self.post_description = "\nReflect on possible errors in the answer above and answer again using the same format. If you think there are no errors in your previous answers that will affect the results, there is no need to correct them.\n"
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str, Dict], temporal_info:Dict[str, Dict], **kwargs):
        query = raw_inputs['query']
        spatial_prompt = message_aggregation(raw_inputs, spatial_info, self.message_aggregation)
        temporal_prompt = message_aggregation(raw_inputs, temporal_info, self.message_aggregation)
        format_prompt = output_format_prompt[self.output_format]
        reason_prompt = reasoning_prompt[self.reason]

        system_prompt = f"{self.description}\n{reason_prompt}"
        system_prompt += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
        user_prompt = f"{query}\n"
        user_prompt += f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}" if spatial_prompt else ""
        user_prompt += f"\n\nIn the last round of dialogue, other agents' outputs were:\n\n{temporal_prompt}" if temporal_prompt else ""
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """
        Run the agent.
        Args:
            inputs: dict[str, str]: Raw inputs.
            spatial_info: dict[str, dict]: Spatial information.
            temporal_info: dict[str, dict]: Temporal information.
        Returns:
            Any: str: Aggregated message.
        """
        query = input['query']
        # passed, response= inner_test(input, spatial_info, temporal_info)
        # if passed:
        #     return response
        # prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        # response = self.llm.gen(prompt)
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        # --- Start of Modification ---
        response = None
        try:
            response = self.llm.gen(prompt)
            if not response:
                logger.error(f"Agent {self.id} (Role: {self.role.role}) received an empty or invalid response from the LLM.")
                return ""
        except Exception as e:
            logger.error(f"Agent {self.id} (Role: {self.role.role}) failed during LLM API call: {e}")
            return ""
        # --- End of Modification --- 
        
        if get_ablation_mode() == 'no_post_process':
            response, validation_info = response, {"passed": True, "errors": []}
        else:
            response, validation_info = post_process(input, response, self.post_process)

        if self.post_process == "CodesysSemanticCheck" and get_ablation_mode() != 'no_post_process':
            max_retries = 1
            for attempt in range(max_retries):
                if validation_info.get("passed", False):
                    break

                logger.warning(f"Agent {self.id} (Role: {self.role.role}) CODESYS validation failed, retrying ({attempt+1}/{max_retries})...")

                error_details = validation_info.get("errors", [])
                error_text = ""
                for e in error_details:
                    desc = e.get("ErrorDesc", str(e))
                    error_text += f"  - {desc}\n"

                retry_system = (
                    f"{self.description}\n"
                    f"\nFix the ST code based on the CODESYS compilation errors below. "
                    f"Output corrected ST code in ```st ... ``` blocks."
                )
                retry_user = (
                    f"## Task\n{query}\n\n"
                    f"## CODESYS Compilation Errors to Fix\n{error_text}\n"
                    f"Please fix ALL compilation errors above and regenerate the correct ST code."
                )
                retry_prompt = [
                    {'role': 'system', 'content': retry_system},
                    {'role': 'user', 'content': retry_user}
                ]

                try:
                    response = self.llm.gen(retry_prompt)
                    if not response:
                        logger.error(f"Agent {self.id} retry returned empty response.")
                        continue
                except Exception as e:
                    logger.error(f"Agent {self.id} retry failed: {e}")
                    continue

                response, validation_info = post_process(input, response, self.post_process)

            if validation_info.get("passed", False):
                if validation_info.get("codesys_used", False):
                    logger.info(f"Agent {self.id} (Role: {self.role.role}) CODESYS validation passed.")
                else:
                    logger.info(f"Agent {self.id} (Role: {self.role.role}) ANTLR syntax check passed (CODESYS unavailable).")
            else:
                if validation_info.get("codesys_used", False):
                    logger.warning(f"Agent {self.id} (Role: {self.role.role}) CODESYS validation failed after all retries.")
                else:
                    logger.warning(f"Agent {self.id} (Role: {self.role.role}) ANTLR syntax check failed after all retries.")

        logger.debug(f"Agent {self.id} Role: {self.role.role} LLM: {self.llm.model_name}")
        logger.debug(f"system prompt:\n {prompt[0]['content']}")
        logger.debug(f"user prompt:\n {prompt[1]['content']}")
        logger.debug(f"response:\n {response}")

        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = f"{query}\nThe initial thinking information is:\n{response} \n Please refer to the new format requirements when replying."
            prompt = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            response = self.llm.gen(prompt)
            logger.debug(f"post system prompt:\n {system_prompt}")
            logger.debug(f"post user prompt:\n {user_prompt}")
            logger.debug(f"post response:\n {response}")
            
            # #! 
            # received_id = []
            # role = self.role.role
            # received_id.append(self.id + '(' + role + ')')

            # entry = {
            #     "id": self.id,
            #     "role": self.role.role,
            #     "llm_name": self.llm.model_name,
            #     "system_prompt": prompt[0]['content'],
            #     "user_prompt": prompt[1]['content'],
            #     "received_id": received_id,
            #     "response": response,
            # }
            # try:
            #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            # except (FileNotFoundError, json.JSONDecodeError):
            #     data = []

            # data.append(entry)

            # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
            #     json.dump(data, f, ensure_ascii=False, indent=2)
            # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return self._execute(input, spatial_info, temporal_info, **kwargs)

@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None, agent_name = "", domain = "", llm_name = "", prompt_file = ""):
        super().__init__(id, agent_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_file = json.load(open(f"{prompt_file}", 'r', encoding='utf-8'))
        self.retry_count = 0
        self.first_pass_code = None

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):  
        system_prompt = f"{self.prompt_file['system']}"
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        acceptance = functional_context(raw_inputs.get("task_id", ""))
        user_prompt = f"The task is:\n\n {raw_inputs['query']}.{acceptance}\n At the same time, the output of other agents is as follows:\n\n{spatial_str} {self.prompt_file['user']}"
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    
    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        response = self.llm.gen(prompt)
        logger.debug(f"Final Refer Node LLM: {self.llm.model_name}")
        logger.debug(f"Final System Prompt:\n {prompt[0]['content']}")
        logger.debug(f"Final User Prompt:\n {prompt[1]['content']}")
        logger.debug(f"Final Response:\n {response}")

        first_code = _extract_st_code(response)
        self.first_pass_code = first_code
        self.retry_count = 0

        if get_ablation_mode() in ('no_codesys_feedback', 'no_post_process'):
            logger.info(f"FinalNode: [Ablation:{get_ablation_mode()}] CODESYS feedback disabled, skipping internal compilation check")
            return response

        return self.repair_existing(input, response)

    def repair_existing(self, input, response):
        """Continue FinalNode validation from an already generated ST candidate."""
        self.first_pass_code = _extract_st_code(response)
        self.retry_count = 0
        self.attempt_records = []

        configured_retries = os.getenv("RL_MAR_MAX_REPAIRS")
        max_retries = int(configured_retries) if configured_retries is not None else (4 if functional_available(input.get("task_id", "")) else 1)
        max_retries = min(max(max_retries, 0), 4)
        repair_mode = os.getenv("RL_MAR_REPAIR_MODE", "all_stages")
        if repair_mode not in {"compile_only", "all_stages"}:
            raise ValueError(f"unknown repair mode: {repair_mode}")
        max_stagnant = int(os.getenv("RL_MAR_MAX_STAGNANT_REPAIRS", "2"))
        if max_stagnant not in {1, 2}:
            raise ValueError(f"unsupported stagnant repair limit: {max_stagnant}")
        best_response = response
        best_score = (-1, -1, -1, -1, -1)
        non_improving_retries = 0
        for attempt in range(1, max_retries + 2):  # attempt 1 = first, 2+ = retries
            code = canonicalize_st(_extract_st_code(response))
            response = replace_st_code(response, code)
            validation_started = time.monotonic()
            executable_result = run_functional_check(input.get("task_id", ""), code)
            validation_seconds = time.monotonic() - validation_started
            score = functional_score(executable_result)
            feedback = functional_feedback(executable_result, input.get("task_id", "")) if executable_result else ""
            record = {
                "attempt": attempt,
                "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                "code": code,
                "status": executable_result.get("status") if executable_result else "UNAVAILABLE",
                "score": list(score),
                "feedback_sha256": hashlib.sha256(feedback.encode("utf-8")).hexdigest(),
                "feedback": feedback,
                "validation_seconds": validation_seconds,
                "validation_summary": public_functional_summary(executable_result) if executable_result else None,
                "selected": False,
                "selection_reason": "candidate did not improve the best score",
            }
            self.attempt_records.append(record)
            if executable_result is not None and executable_result.get("status") == "DATASET_ERROR":
                logger.warning(f"FinalNode: dataset definition error for {input.get('task_id', '')}; skipping retries.")
                return response
            if executable_result is not None:
                if score > best_score:
                    best_score = score
                    best_response = response
                    non_improving_retries = 0
                    for previous in self.attempt_records:
                        if previous.get("selected"):
                            previous["selection_reason"] = f"superseded by attempt {attempt}"
                        previous["selected"] = False
                    record["selected"] = True
                    record["selection_reason"] = "highest lexicographic functional score"
                elif attempt > 1:
                    non_improving_retries += 1
                if executable_result.get("status") == "PASS":
                    record["stop_reason"] = "validation_passed"
                    logger.info(f"FinalNode: executable checks passed (attempt {attempt}).")
                    return response
                if repair_mode == "compile_only" and executable_result.get("status") not in {"COMPILE_FAIL", "COMPILE_ERROR"}:
                    record["stop_reason"] = "repair_mode_compile_only"
                    return best_response
                validation_label = "Executable Validation"
                error_text = functional_feedback(executable_result, input.get("task_id", ""))
            else:
                validation_label = "CODESYS Compilation"
                error_text = ""
            try:
                if executable_result is not None and executable_result.get("status") != "DATASET_ERROR":
                    codesys_passed = False
                    codesys_result = {"errors": [{"ErrorDesc": error_text}]}
                else:
                    from MAR.Tools.PLC.codesys_client import get_codesys_client
                    client = get_codesys_client()
                    codesys_result = client.compile_code(code)
                    codesys_passed = codesys_result.get("success", False)
            except Exception as e:
                logger.warning(f"FinalNode: CODESYS client error (attempt {attempt}): {e}")
                codesys_passed = False
                codesys_result = {"errors": [{"ErrorDesc": str(e)}]}

            if codesys_passed:
                logger.info(f"FinalNode: CODESYS compilation passed (attempt {attempt}).")
                return response

            error_count = len(codesys_result.get("errors", []))
            logger.warning(f"FinalNode: {validation_label} failed (attempt {attempt}/{max_retries + 1}), {error_count} error(s).")

            if attempt > max_retries:
                record["stop_reason"] = "repair_budget_exhausted"
                logger.warning(f"FinalNode: max retries ({max_retries + 1}) exhausted, returning last response.")
                return best_response if best_score > (-1, -1) else response

            if non_improving_retries >= max_stagnant:
                record["stop_reason"] = "non_improving_limit"
                logger.warning(f"FinalNode: stopping after {max_stagnant} consecutive non-improving retries.")
                return best_response

            error_text = sanitize_compiler_diagnostic("\n".join(
                f"  - {e.get('ErrorDesc', str(e))}"
                for e in codesys_result.get("errors", [])
            ))

            retry_system = (
                f"{self.prompt_file.get('system', '')}\n\n"
                f"Your previous ST code failed {validation_label} with the following errors.\n"
                f"{functional_repair_instructions()}\n"
                f"Fix ALL of them and output the corrected ST code in ```st ... ``` blocks."
            )
            retry_user = (
                f"## Task\n{input.get('query', '')}\n\n"
                f"## Executable Contract\n{functional_context(input.get('task_id', ''))}\n\n"
                f"## Previous ST Code\n```st\n{code}\n```\n\n"
                f"## {validation_label} Errors to Fix\n{error_text}\n"
                f"Please fix all reported problems and regenerate the complete correct ST code."
            )
            retry_prompt = [
                {'role': 'system', 'content': retry_system},
                {'role': 'user', 'content': retry_user}
            ]

            self.retry_count += 1
            logger.debug(f"FinalNode retry {self.retry_count} response:\n{response}")
            try:
                generation_started = time.monotonic()
                response = self.llm.gen(retry_prompt)
                if not response:
                    logger.error("FinalNode retry returned empty response.")
                    self.attempt_records.append({
                        "attempt": attempt + 1,
                        "status": "GENERATION_ERROR",
                        "diagnostic": "empty LLM response",
                        "generation_seconds": time.monotonic() - generation_started,
                        "selected": False,
                    })
                    return best_response if best_score > (-1, -1) else response
            except Exception as e:
                logger.error(f"FinalNode retry LLM call failed: {e}")
                self.attempt_records.append({
                    "attempt": attempt + 1,
                    "status": "GENERATION_ERROR",
                    "diagnostic": str(e),
                    "generation_seconds": time.monotonic() - generation_started,
                    "selected": False,
                })
                return best_response if best_score > (-1, -1) else response

        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return self._execute(input, spatial_info, temporal_info, **kwargs)
