from __future__ import annotations

import argparse

from workflow.demo.linear.workflow import LinearWorkflowInput, build_linear_demo
from workflow.demo.runner import run_single as run_single_workflow
from workflow.demo.runner import run_traffic as run_workflow_traffic


def _build_demo(args):
    return build_linear_demo(
        researcher_max_tokens=args.researcher_max_tokens,
        reasoner_max_tokens=args.reasoner_max_tokens,
        extractor_max_tokens=args.extractor_max_tokens,
        temperature=args.temperature,
    )


def _build_input(_: argparse.Namespace, question: str, context: str) -> LinearWorkflowInput:
    return LinearWorkflowInput(question=question, context=context)


def run_single(args) -> None:
    run_single_workflow(args, build_demo=_build_demo, build_input=_build_input)


def run_traffic(args) -> None:
    run_workflow_traffic(args, build_demo=_build_demo, build_input=_build_input)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the linear workflow demo on top of the workflow runtime."
    )
    parser.add_argument("--mode", choices=("single", "traffic"), default="single")
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
    parser.add_argument("--question_prefix", type=str, default="Demo question")
    parser.add_argument("--researcher_max_tokens", type=int, default=192)
    parser.add_argument("--reasoner_max_tokens", type=int, default=192)
    parser.add_argument("--extractor_max_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_inflight_requests", type=int, default=0)
    parser.add_argument("--total_workflows", type=int, default=4)
    parser.add_argument("--target_concurrency", type=int, default=2)
    parser.add_argument("--print_history", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_inflight_requests <= 0:
        args.max_inflight_requests = None

    if args.mode == "single":
        run_single(args)
        return
    run_traffic(args)


if __name__ == "__main__":
    main()
