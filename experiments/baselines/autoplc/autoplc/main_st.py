from dotenv import load_dotenv
load_dotenv()

from common import Config
from common.generate_summary import generate_experiment_summary
import logging
from rich.logging import RichHandler
# 初始化 logger
logger = logging.getLogger("autoplc_st")
logger.setLevel(logging.INFO)

# 仅添加 Rich 控制台输出，不写文件
console_handler = RichHandler(rich_tracebacks=True, markup=True)

logger.handlers.clear()  # 清除旧 handler（避免重复）
logger.addHandler(console_handler)

from autoplc_st import run_autoplc_st
import argparse
if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="oscat", help="benchmark name")
    parser.add_argument("--config", type=str, default="default_st", help="config name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint where the task is saved")
    parser.add_argument("--single_task", type=int, default=None, help="run only the N-th task (1-indexed)")
    parser.add_argument("--task_range", type=int, nargs=2, default=None, metavar=("START", "END"), help="run tasks from START to END (1-indexed, inclusive)")
    args = parser.parse_args()

    if args.single_task is not None and args.task_range is not None:
        parser.error("--single_task and --task_range cannot be used together")

    cfg = Config(config_file=args.config)
    base_folder = run_autoplc_st(
        benchmark=args.benchmark, config=cfg,
        checkpoint_dir=args.checkpoint_dir, max_workers=1,
        single_task=args.single_task,
        task_range=args.task_range,
    )

    # Post-processing: generate llm4plc_baseline-compatible summary files
    if base_folder:
        benchmark_path = str(cfg.ROOTPATH / f"data/benchmarks/{args.benchmark}.jsonl")
        generate_experiment_summary(
            base_folder=base_folder,
            benchmark_file=benchmark_path,
            config=cfg,
            method="autoplc_st",
            model_name=cfg.model,
        )
