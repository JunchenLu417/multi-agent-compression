from __future__ import annotations

import argparse

from workflow.demo.fanout.workflow import DagWorkflowInput, build_dag_demo
from workflow.demo.runner import run_single as run_single_workflow


def _build_demo(args):
    return build_dag_demo(
        planner_max_tokens=args.planner_max_tokens,
        branch_max_tokens=args.branch_max_tokens,
        synth_max_tokens=args.synth_max_tokens,
        temperature=args.temperature,
    )


def _build_input(_: argparse.Namespace, question: str, context: str) -> DagWorkflowInput:
    return DagWorkflowInput(question=question, context=context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fan-out workflow demo on top of the workflow runtime."
    )
    parser.add_argument("--model_path", type=str, default="", help="Model path for real Sparse-vLLM execution.")
    parser.add_argument("--vllm_sparse_method", type=str, default="", help="Sparse method passed to Sparse-vLLM.")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--hyper_params", type=str, default="{}", help="Extra LLM kwargs as a JSON object.")
    parser.add_argument("--question", type=str, default="What is the best rollout plan for this feature?")
    parser.add_argument(
        "--context",
        type=str,
        default=(
            "The feature adds a new workflow runtime. The team wants a staged rollout, strong observability, "
            "and explicit risk handling before production traffic is raised."
        ),
    )
    parser.add_argument("--planner_max_tokens", type=int, default=192)
    parser.add_argument("--branch_max_tokens", type=int, default=192)
    parser.add_argument("--synth_max_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target_concurrency", type=int, default=1)
    parser.add_argument("--max_inflight_requests", type=int, default=0)
    parser.add_argument("--print_history", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_inflight_requests <= 0:
        args.max_inflight_requests = None

    run_single_workflow(args, build_demo=_build_demo, build_input=_build_input)


if __name__ == "__main__":
    main()
