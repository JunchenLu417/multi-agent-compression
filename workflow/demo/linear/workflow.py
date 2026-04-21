from __future__ import annotations

from dataclasses import dataclass

from sparsevllm import SamplingParams

from workflow.handlers import HandlerRegistry, WorkflowNodeHandler
from workflow.spec import EdgeSpec, ExecutionLimits, FinishPolicy, HistorySpec, NodeSpec, TriggerSpec, WorkflowSpec
from workflow.types import NodeRequest, NodeResult


@dataclass(frozen=True)
class LinearWorkflowInput:
    question: str
    context: str


# System Prompt
RESEARCHER_SYSTEM = (
    "You are a research analyst. Extract all relevant evidence from the provided context. "
    "Be comprehensive because downstream agents depend on your notes."
)
REASONER_SYSTEM = (
    "You are a reasoning specialist. Use the existing conversation history to derive the answer "
    "step by step without dropping important evidence."
)
EXTRACTOR_SYSTEM = (
    "You are an answer extractor. Use the prior conversation history to produce the final answer "
    "as concisely as possible."
)


# Human Message
def researcher_inst(payload: LinearWorkflowInput) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        "Please extract all relevant facts and evidence."
    )


def reasoner_inst(payload: LinearWorkflowInput, researcher_output: str) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Researcher output:\n{researcher_output}\n\n"
        "Using the question, context, and researcher output above, reason step by step to answer the question."
    )


def extractor_inst(
    payload: LinearWorkflowInput,
    researcher_output: str,
    reasoner_output: str,
) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Researcher output:\n{researcher_output}\n\n"
        f"Reasoner output:\n{reasoner_output}\n\n"
        "Using the question, context, and prior agent outputs above, extract the final answer concisely."
    )


def _state_text(state_data, key: str) -> str:
    value = state_data.get(key, "")
    return value if isinstance(value, str) else str(value)


# Handler: per-agent behavior object
class ResearcherHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: LinearWorkflowInput = context.workflow_input
        return NodeRequest.from_current_prompt(
            prompt=researcher_inst(payload),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"researcher_output": output.text},
            state_delta={"researcher_output": output.text},
        )


class ReasonerHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: LinearWorkflowInput = context.workflow_input
        researcher_output = _state_text(context.state_data, "researcher_output")
        return NodeRequest.from_current_prompt(
            prompt=reasoner_inst(payload, researcher_output),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"reasoner_output": output.text},
            state_delta={"reasoner_output": output.text},
        )


class ExtractorHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: LinearWorkflowInput = context.workflow_input
        researcher_output = _state_text(context.state_data, "researcher_output")
        reasoner_output = _state_text(context.state_data, "reasoner_output")
        return NodeRequest.from_current_prompt(
            prompt=extractor_inst(payload, researcher_output, reasoner_output),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"final_answer": output.text},
            state_delta={"final_answer": output.text},
        )


def build_linear_demo(
    *,
    researcher_max_tokens: int = 192,
    reasoner_max_tokens: int = 192,
    extractor_max_tokens: int = 192,
    temperature: float = 0.0,
) -> tuple[WorkflowSpec, HandlerRegistry]:
    workflow = WorkflowSpec(
        name="research_chain_demo",
        nodes=(
            NodeSpec(
                id="researcher",
                handler="researcher",
                trigger=TriggerSpec.entry(),
                history=HistorySpec(system_prompt=RESEARCHER_SYSTEM, include_ancestor_history=False),
            ),
            NodeSpec(
                id="reasoner",
                handler="reasoner",
                trigger=TriggerSpec.all_of("researcher"),
                history=HistorySpec(system_prompt=REASONER_SYSTEM, include_ancestor_history=False),
            ),
            NodeSpec(
                id="extractor",
                handler="extractor",
                trigger=TriggerSpec.all_of("reasoner"),
                history=HistorySpec(system_prompt=EXTRACTOR_SYSTEM, include_ancestor_history=False),
            ),
        ),
        edges=(
            EdgeSpec("researcher", "reasoner"),
            EdgeSpec("reasoner", "extractor"),
        ),
        finish_policy=FinishPolicy.node("extractor"),
        limits=ExecutionLimits(
            max_total_firings=6,
            max_firings_per_node={"researcher": 1, "reasoner": 1, "extractor": 1},
        ),
        description="Three-agent linear workflow using per-node system prompts without ancestor history accumulation.",
    )

    handlers = HandlerRegistry(
        {
            "researcher": ResearcherHandler(max_tokens=researcher_max_tokens, temperature=temperature),
            "reasoner": ReasonerHandler(max_tokens=reasoner_max_tokens, temperature=temperature),
            "extractor": ExtractorHandler(max_tokens=extractor_max_tokens, temperature=temperature),
        }
    )
    return workflow, handlers


ResearchWorkflowInput = LinearWorkflowInput
build_research_demo = build_linear_demo


__all__ = [
    "LinearWorkflowInput",
    "ResearchWorkflowInput",
    "build_linear_demo",
    "build_research_demo",
]
