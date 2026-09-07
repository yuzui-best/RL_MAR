import os
import shutil
import sys
import time
import json
from autoplc_st.agents.clients import ClientManager
from autoplc_st.tools import APIDataLoader, PromptResultUtil
import multiprocessing
import shutil
import concurrent.futures
from functools import partial
from autoplc_st.agents import (
    Retriever,
    Modeler,
    ApiAgent,
    LogicComposer,
    AutoDebugger,
    LearnAgent,
    init_team_log_path
)
from common import Config, ROOTPATH
from autoplc_st.tools.codesys_debugger import CompilerInfrastructureError
import logging
logger = logging.getLogger("autoplc_st")


root_path = ROOTPATH
def generate_plans():
    from autoplc_st.agents.plan_gen import gen_plan_dataset
    global root_path
    # root_path = os.getenv("ROOTPATH")
    case_requirement_dir = os.path.join(root_path, "data", "rag_data", "st", "st_case_requirement")
    # case_requirement_dir = os.path.join(root_path, "src", "experiment", "datasets", "competition")
    case_code_dir = os.path.join(root_path, "data", "rag_data", "st", "st_case_code")
    case_plan_dir = os.path.join(root_path, "data", "rag_data", "st", "st_case_plan_experiment")
    gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir)


def decide_is_state_machine_in_lgf():
    from autoplc_st.agents.plan_gen import figure_state_machine_in_lgf
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data", "benchmarks", "lgf_state_machine.jsonl")
    figure_state_machine_in_lgf(dataset_file)

def baseline_in_github_case(model):
    from autoplc_st.agents.plan_gen import run_baseline_in_github_case
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data", "benchmarks", "githubcase.jsonl")
    run_baseline_in_github_case(model, dataset_file)

def run_autoplc_st(benchmark: str, config: Config, checkpoint_dir: str = None, max_workers: int = 1, single_task: int = None, task_range: tuple = None):
    global root_path
    root_path = ROOTPATH
    benchmark_file_path = os.path.join(root_path, "data", "benchmarks", f"{benchmark}.jsonl")
    if not os.path.exists(benchmark_file_path):
        logger.error(f"Benchmark file {benchmark_file_path} not found.")
        return

    if checkpoint_dir:
        ## 从 checkpoint_dir 中加载配置
        config_path = os.path.join(checkpoint_dir, "config", "config.yaml")
        if os.path.exists(config_path):
            config = Config.load_from_absolute_path(config_path)
            logger.info(f"Loaded config from {config_path}.")
        else:
            raise ValueError(f"Checkpoint config file {config_path} not found.")

    # 输出config关键内容：
    logger.info(f"Benchmark:{benchmark} Model: {config.model} retriever: {config.RETRIEVE_DISABLED == False} planner: {config.MODELING_DISABLED == False} api_rec: {config.APIREC_DISABLED == False} debugger: {config.DEBUGGER_DISABLED == False} auto_learn: {config.AUTOLEARN_DISABLED == False}")

    ClientManager().set_config(config)
    APIDataLoader.init_load(config = config)
    base_folder_old = init_team_log_path(checkpoint_dir)
    base_folder = base_folder_old

    if not checkpoint_dir:
        # 重命名为 llm4plc 风格: {model_slug}_{timestamp}/
        model_slug = config.model.replace("/", "-").replace(":", "-")
        tz = __import__('pytz').timezone('Asia/Shanghai')
        ts = __import__('datetime').datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        new_dir = os.path.join(os.path.dirname(str(base_folder_old)), f"{model_slug}_{ts}")
        if str(base_folder_old) != new_dir:
            # 移动旧目录内容到新目录
            os.makedirs(new_dir, exist_ok=True)
            for item in os.listdir(str(base_folder_old)):
                shutil.move(os.path.join(str(base_folder_old), item), os.path.join(new_dir, item))
            os.rmdir(str(base_folder_old))
            # 更新所有 agent 的引用
            from autoplc_st.agents import Retriever, Modeler, LogicComposer, AutoDebugger
            for cls in [Retriever, Modeler, LogicComposer, AutoDebugger]:
                cls.base_logs_folder = new_dir
            base_folder = new_dir

        # 保存配置文件
        os.makedirs(os.path.join(base_folder, "config"), exist_ok=True)
        shutil.copy(config.config_path, os.path.join(base_folder, "config", "config.yaml"))

    # log 输出到 logs/{model_timestamp}/ 下，不同实验不混在一起
    exp_log_folder = os.path.basename(base_folder)  # deepseek-ai-DeepSeek-V3.2_20260507_180738
    ClientManager().set_logs_folder(exp_log_folder)

    all_agents_start_time = time.time()
    task_timings = {}

    with open(benchmark_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        all_tasks = [json.loads(line) for line in lines]

    # single_task / task_range 过滤（1-indexed）
    if single_task is not None:
        if single_task < 1 or single_task > len(all_tasks):
            logger.error(f"single_task {single_task} out of range (1-{len(all_tasks)})")
            return
        tasks = [all_tasks[single_task - 1]]
        logger.info(f"Single task mode: running only task {single_task}/{len(all_tasks)}: {tasks[0]['name']}")
    elif task_range is not None:
        start, end = task_range
        if start < 1 or end > len(all_tasks) or start > end:
            logger.error(f"task_range [{start}, {end}] out of range (1-{len(all_tasks)})")
            return
        tasks = all_tasks[start - 1:end]
        logger.info(f"Task range mode: running tasks {start} to {end} ({len(tasks)} tasks)")
    else:
        tasks = all_tasks
        logger.info(f"Full dataset mode: {len(tasks)} tasks")

    # 多线程执行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建偏函数，固定base_folder和config参数
        task_processor = partial(
            _process_task,
            base_folder=base_folder,
            config=config,
            logger=logger
        )

        # 提交所有任务
        future_to_task = {executor.submit(task_processor, task, idx): (task, idx) for idx, task in enumerate(tasks) }

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()  # 获取任务结果（如果有异常会在这里抛出）
            except Exception as e:
                logger.error(f"Unexpected error processing task {task['name']}: {str(e)}")

    total_time = time.time() - all_agents_start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds.")

    # 收集每个任务的执行时间（从各任务目录下的 _timing.json 读取）
    for task in tasks:
        name = task["name"]
        timing_path = os.path.join(base_folder, name, "_timing.json")
        if os.path.exists(timing_path):
            try:
                with open(timing_path, "r", encoding="utf-8") as f:
                    task_timings[name] = json.load(f).get("execution_time", 0)
            except Exception:
                task_timings[name] = 0

    # 保存总体实验计时
    timing_file = os.path.join(base_folder, "_experiment_timing.json")
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_start_time": all_agents_start_time,
            "total_execution_time": round(total_time, 2),
            "task_timings": task_timings
        }, f, indent=2)
    logger.info(f"Timing data saved to {timing_file}")

    # Return base_folder for post-processing by main_st.py
    return base_folder


def _process_task(task: dict, idx, base_folder: str, config: Config, logger):
    task_name = task["name"]
    task_path = os.path.join(base_folder, task_name)
    
    # 检查任务是否已完成
    if os.path.exists(task_path):
        logger.info(f"Task {task_name} already completed. Skipping...")
        return
    
    try:

        ip_port = config.CODESYS_IP[idx % len(config.CODESYS_IP)]
        # 执行任务工作流（注意这里调用的是autoplc_st_workflow）
        autoplc_st_workflow(task, base_folder, config, ip_port)
    except CompilerInfrastructureError:
        raise
    except Exception as e:
        logger.error(f"Failed to process task {task_name}: {str(e)}")
        raise  # 重新抛出异常，让主函数知道任务失败

def autoplc_st_workflow(
        task: dict, 
        base_folder: str,
        config:Config,
        ip_port: str
    ):

    # 检查是否存在对应的 st 文件
    st_file_path = os.path.join(config.ST_CODE_DIR, f"{task['name']}.st")
    if os.path.exists(st_file_path):
        groundtruth_st = open(st_file_path, "r", encoding="utf8").read()
        logger.info(f"Loaded groundtruth st: {task['name']}")
    else:
        groundtruth_st = None
        logger.warning(f"Groundtruth st for {task['name']} not found.")


    # 获取大模型客户端
    openai_client = ClientManager().get_openai_client()
    zhipuai_client = ClientManager().get_zhipuai_client()
    local_api_retriever = ClientManager().get_local_api_retriever()

    os.makedirs(os.path.join(base_folder, f"{task['name']}"), exist_ok=True)

    start_time = time.time()

    # 初始化finally块中的变量
    sample_names = []
    logic_for_this_task = ""
    apis_for_this_task = []

    import traceback
    try:
        ###############    retrieve examples    ################
        retrieved_samples = []
        if not config.RETRIEVE_DISABLED:
            retrieved_samples = Retriever.run_retrieve_case(
                task=task,
                alternatives=config.CASE_ALTERNATIVES,
                zhipuai_client=zhipuai_client
            )
            

        ################ load related algorithms and possible apis ################
        sample_names = [sample["name"] for sample in retrieved_samples]
        logger.info(f"Retrieved samples: {sample_names}")

        related_algorithm = []
        for name in sample_names:
            # algo = PromptResultUtil.get_plan(name=name, code_type="st")      #为了跑不同版本的plan 需要传plan的版本的名称
            algo = PromptResultUtil.get_plan_diff(
                name=name, 
                plan_version = config.PLAN_SET_NAME
            )
            if algo is not None:
                related_algorithm.append(algo)
            else:
                related_algorithm.append("")

        if sample_names:
            #获取相似案例用到的api
            api_from_similar_cases = APIDataLoader.extract_apis_from_cases(sample_names)    
        else:
            api_from_similar_cases = []

        ################      generate plan      ################
        logic_for_this_task = "NOT PROVIDED"
        if not config.MODELING_DISABLED:
            logic_for_this_task = Modeler.run_modeling_task(
                task=task, 
                retrieved_examples=retrieved_samples,
                related_algorithm=related_algorithm,
                openai_client=openai_client,
                load_few_shots=config.IS_MODELING_FEWSHOT
            )
            # print(f"[INFO] logic for this task:\n {logic_for_this_task}")
            # save logic for this task to file
        else:
            related_algorithm = ["NOT PROVIDED"] * len(retrieved_samples)
        
        ################     api recommend      ################
        if not config.APIREC_DISABLED:
            api_recommend, library_func_recommend  = ApiAgent.run_recommend_api(
                task=task, 
                algorithm_for_this_task=logic_for_this_task,
                openai_client=openai_client,
                zhipuai_client=zhipuai_client,
                local_api_retriever=local_api_retriever,
                load_few_shots=config.IS_APIREC_FEWSHOT
            )
        else:
            api_recommend = []
            library_func_recommend = []
            
        apis_for_this_task = list(set(api_recommend + api_from_similar_cases))

        ################      generate st      ################
        st_code = LogicComposer.run_gen_st(
            task=task,
            retrieved_examples = retrieved_samples,
            related_algorithm = related_algorithm,
            logic_for_this_task= logic_for_this_task,
            apis_for_this_task = apis_for_this_task,
            openai_client = openai_client,
            load_few_shots=config.IS_CODING_FEWSHOT
        )
        first_gen_st = st_code

        ################       debug st      ################
        if not config.DEBUGGER_DISABLED:
            st_code = AutoDebugger.run_debugger_with_compiler(
                task=task,
                st_code=st_code,
                max_verify_count=config.VERIFY_COUNT,
                openai_client=openai_client,
                load_few_shots=config.IS_DEBUGGER_FEWSHOT,
                ip_port = ip_port
            )
        else:
            # save st code to file
            code_output_file = os.path.join(base_folder, f"{task['name']}/{task['name']}_{0}.st")
            logger.info(f"output file is {code_output_file}")
            with open(code_output_file, "w", encoding="utf-8") as fp:
                fp.write(st_code)
        
        ###############    auto learner      ################
        if not config.AUTOLEARN_DISABLED:

            # 加载history
            if os.path.exists(os.path.join(base_folder, f"{task['name']}", "verify_info.jsonl")):
                try:
                    with open(os.path.join(base_folder, f"{task['name']}", "verify_info.jsonl"), "r", encoding="utf-8") as f:
                        debug_history = [json.loads(line) for line in f.readlines()]
                except Exception as e:
                    logger.error(f"Error decoding JSON from verify_info.jsonl: {e}")
                    debug_history = []
            else:
                debug_history = []

            if groundtruth_st is not None:
                logger.info("Start auto learner from groundtruth st.")
                coding_feed_back = LearnAgent.run_learn_from_coding(
                    task=task,
                    prediction_st=first_gen_st, # 这里用的是第一次生成的st代码，因为我们希望模型提高首次生成效率
                    openai_client=openai_client,
                    groundtruth_st=groundtruth_st,
                    debug_history=debug_history
                )
                # save feedback json
                with open(os.path.join(base_folder, f"{task['name']}", "coding_feedback.json"), "w", encoding="utf-8") as f:
                    json.dump(coding_feed_back, f, ensure_ascii=False, indent=4)

            

            if len(debug_history) > 0 and groundtruth_st is not None:
                logger.info("Start auto learner from debug history.")
                debug_feed_back = LearnAgent.run_learn_from_debug(
                    task = task,
                    groundtruth_st = groundtruth_st,
                    debug_history = debug_history,
                    openai_client = openai_client
                )
                # save debug feedback json
                with open(os.path.join(base_folder, f"{task['name']}", "debug_feedback.json"), "w", encoding="utf-8") as f:
                    json.dump(debug_feed_back, f, ensure_ascii=False, indent=4)


    except Exception as e:
        logger.exception(f"Error occurred while processing task {task['name']}: {e}")
        logger.exception(e)
    finally :
        # save all intermediate results to file
        try:
            with open(os.path.join(base_folder, f"{task['name']}", "intermediate_results.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "retrieved_samples": sample_names if sample_names else [],
                    "logic_for_this_task": logic_for_this_task if logic_for_this_task else "",
                    "apis_for_this_task": apis_for_this_task if apis_for_this_task else [],
                },fp=f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error occurred while saving intermediate results for task {task['name']}: {e}")
            pass

        # 保存该任务的执行时间
        task_elapsed = round(time.time() - start_time, 2)
        try:
            with open(os.path.join(base_folder, task['name'], "_timing.json"), "w", encoding="utf-8") as f:
                json.dump({"execution_time": task_elapsed}, f)
        except Exception:
            pass

    logger.info(f"Task {task['name']} completed in {time.time() - start_time:.2f} seconds.")
