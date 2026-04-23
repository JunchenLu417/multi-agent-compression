from __future__ import annotations

from dataclasses import dataclass

from sparsevllm import SamplingParams

from workflow.handlers import HandlerRegistry, WorkflowNodeHandler
from workflow.spec import EdgeSpec, ExecutionLimits, FinishPolicy, HistorySpec, NodeSpec, TriggerSpec, WorkflowSpec
from workflow.types import NodeRequest, NodeResult


@dataclass(frozen=True)
class DagWorkflowInput:
    question: str
    context: str


PLANNER_SYSTEM = (
    "You are a planning agent. Read the task and outline the key dimensions that downstream "
    "specialists should analyze."
)
SUPPORT_SYSTEM = (
    "You are a supporting-evidence agent. Use the provided question, context, and planner output to surface the "
    "strongest arguments in favor of the proposed answer."
)
RISK_SYSTEM = (
    "You are a risk-analysis agent. Use the provided question, context, and planner output to surface the strongest "
    "risks, caveats, and missing checks."
)
SYNTH_SYSTEM = (
    "You are a synthesis agent. Merge the provided planner, support, and risk outputs into one final answer that balances "
    "the supporting arguments and the risks."
)


def _planner_prompt(payload: DagWorkflowInput) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        "Create a short plan for how to analyze this question."
    )


def _support_prompt(payload: DagWorkflowInput, plan: str) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Planner output:\n{plan}\n\n"
        "Using the question, context, and planner output above, list the strongest supporting arguments and favorable evidence."
    )


def _risk_prompt(payload: DagWorkflowInput, plan: str) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Planner output:\n{plan}\n\n"
        "Using the question, context, and planner output above, list the main risks, caveats, and checks that could invalidate the answer."
    )


def _synth_prompt(
    payload: DagWorkflowInput,
    plan: str,
    supporting_points: str,
    risk_points: str,
) -> str:
    return (
        f"Question: {payload.question}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Planner output:\n{plan}\n\n"
        f"Support output:\n{supporting_points}\n\n"
        f"Risk output:\n{risk_points}\n\n"
        "Using the question, context, and prior agent outputs above, produce the final answer and explicitly balance upside against risk."
    )


def _state_text(state_data, key: str) -> str:
    value = state_data.get(key, "")
    return value if isinstance(value, str) else str(value)


class PlannerHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: DagWorkflowInput = context.workflow_input
        return NodeRequest.from_current_prompt(
            prompt=_planner_prompt(payload),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"plan": output.text},
            state_delta={"plan": output.text},
        )


class SupportHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: DagWorkflowInput = context.workflow_input
        plan = _state_text(context.state_data, "plan")
        return NodeRequest.from_current_prompt(
            prompt=_support_prompt(payload, plan),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"supporting_points": output.text},
            state_delta={"supporting_points": output.text},
        )


class RiskHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: DagWorkflowInput = context.workflow_input
        plan = _state_text(context.state_data, "plan")
        return NodeRequest.from_current_prompt(
            prompt=_risk_prompt(payload, plan),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"risk_points": output.text},
            state_delta={"risk_points": output.text},
        )


class SynthHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: DagWorkflowInput = context.workflow_input
        plan = _state_text(context.state_data, "plan")
        supporting_points = _state_text(context.state_data, "supporting_points")
        risk_points = _state_text(context.state_data, "risk_points")
        return NodeRequest.from_current_prompt(
            prompt=_synth_prompt(payload, plan, supporting_points, risk_points),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"final_answer": output.text},
            state_delta={"final_answer": output.text},
        )


def build_dag_demo(
    *,
    planner_max_tokens: int = 192,
    branch_max_tokens: int = 192,
    synth_max_tokens: int = 192,
    temperature: float = 0.0,
) -> tuple[WorkflowSpec, HandlerRegistry]:
    workflow = WorkflowSpec(
        name="dag_fanout_join_demo",
        nodes=(
            NodeSpec(
                id="planner",
                handler="planner",
                trigger=TriggerSpec.entry(),
                history=HistorySpec(system_prompt=PLANNER_SYSTEM, include_ancestor_history=False),
            ),
            NodeSpec(
                id="support",
                handler="support",
                trigger=TriggerSpec.all_of("planner"),
                history=HistorySpec(system_prompt=SUPPORT_SYSTEM, include_ancestor_history=False),
            ),
            NodeSpec(
                id="risk",
                handler="risk",
                trigger=TriggerSpec.all_of("planner"),
                history=HistorySpec(system_prompt=RISK_SYSTEM, include_ancestor_history=False),
            ),
            NodeSpec(
                id="synth",
                handler="synth",
                trigger=TriggerSpec.all_of("support", "risk"),
                history=HistorySpec(system_prompt=SYNTH_SYSTEM, include_ancestor_history=False),
            ),
        ),
        edges=(
            EdgeSpec("planner", "support"),
            EdgeSpec("planner", "risk"),
            EdgeSpec("support", "synth"),
            EdgeSpec("risk", "synth"),
        ),
        finish_policy=FinishPolicy.node("synth"),
        limits=ExecutionLimits(
            max_total_firings=4,
            max_firings_per_node={"planner": 1, "support": 1, "risk": 1, "synth": 1},
        ),
        description="Fan-out / fan-in DAG demo using explicit state passing instead of ancestor history accumulation.",
    )

    handlers = HandlerRegistry(
        {
            "planner": PlannerHandler(max_tokens=planner_max_tokens, temperature=temperature),
            "support": SupportHandler(max_tokens=branch_max_tokens, temperature=temperature),
            "risk": RiskHandler(max_tokens=branch_max_tokens, temperature=temperature),
            "synth": SynthHandler(max_tokens=synth_max_tokens, temperature=temperature),
        }
    )
    return workflow, handlers
