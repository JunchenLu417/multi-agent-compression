from __future__ import annotations

from dataclasses import dataclass

from sparsevllm import SamplingParams

from workflow.handlers import HandlerRegistry, WorkflowNodeHandler
from workflow.spec import EdgeSpec, ExecutionLimits, FinishPolicy, HistorySpec, NodeSpec, TriggerSpec, WorkflowSpec
from workflow.types import NodeRequest, NodeResult


@dataclass(frozen=True)
class LoopWorkflowInput:
    task: str
    context: str


WRITER_SYSTEM = (
    "You are a drafting agent. Use the provided task, context, prior draft, and reviewer feedback "
    "to write the best current draft."
)
REVIEWER_SYSTEM = (
    "You are a review agent. Evaluate the provided task, context, and latest draft. Start your response with either "
    "'ROUTE: revise' or 'ROUTE: approve', then explain your reasoning briefly."
)
FINAL_SYSTEM = (
    "You are a finalizer. Use the provided task, context, latest draft, and reviewer feedback to write the final answer only."
)


def _initial_writer_prompt(payload: LoopWorkflowInput) -> str:
    return (
        f"Task: {payload.task}\n\n"
        f"Context:\n{payload.context}\n\n"
        "Write the initial draft."
    )


def _revision_writer_prompt(payload: LoopWorkflowInput, draft: str, review_feedback: str) -> str:
    return (
        f"Task: {payload.task}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Current draft:\n{draft}\n\n"
        f"Reviewer feedback:\n{review_feedback}\n\n"
        "Revise the draft using the task, context, current draft, and reviewer feedback above."
    )


def _review_prompt(payload: LoopWorkflowInput, draft: str) -> str:
    return (
        f"Task: {payload.task}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Latest draft:\n{draft}\n\n"
        "Review the latest draft. If it still needs work, start with 'ROUTE: revise'. "
        "If it is ready, start with 'ROUTE: approve'. Then explain why."
    )


def _final_prompt(payload: LoopWorkflowInput, draft: str, review_feedback: str) -> str:
    return (
        f"Task: {payload.task}\n\n"
        f"Context:\n{payload.context}\n\n"
        f"Approved draft:\n{draft}\n\n"
        f"Reviewer feedback:\n{review_feedback}\n\n"
        "Using the task, context, approved draft, and reviewer feedback above, produce the final answer only."
    )


def _state_text(state_data, key: str) -> str:
    value = state_data.get(key, "")
    return value if isinstance(value, str) else str(value)


def _parse_route(output_text: str) -> str:
    for line in output_text.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("route:"):
            candidate = normalized.split(":", 1)[1].strip()
            if candidate in {"revise", "approve"}:
                return candidate
    return "revise"


class WriterHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: LoopWorkflowInput = context.workflow_input
        draft = _state_text(context.state_data, "draft")
        review_feedback = _state_text(context.state_data, "review_feedback")
        prompt = (
            _revision_writer_prompt(payload, draft, review_feedback)
            if review_feedback.strip()
            else _initial_writer_prompt(payload)
        )
        return NodeRequest.from_current_prompt(
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"draft": output.text},
            state_delta={"draft": output.text},
        )


class ReviewerHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float, max_firings: int):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_firings = int(max_firings)

    def build_request(self, context) -> NodeRequest:
        payload: LoopWorkflowInput = context.workflow_input
        draft = _state_text(context.state_data, "draft")
        return NodeRequest.from_current_prompt(
            prompt=_review_prompt(payload, draft),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        route = _parse_route(output.text)
        review_feedback = output.text
        # force to finalization if max_firings reached
        forced_finalization = route == "revise" and int(context.firing_index) >= self.max_firings
        if forced_finalization:
            route = "approve"
            review_feedback = (
                f"{output.text}\n\n"
                f"System note: max reviewer firings ({self.max_firings}) reached; routing to finalizer."
            )
        return NodeResult(
            route=route,
            # test max firings
            # route = route if forced_finalization else "revise",
            artifacts={"review_feedback": review_feedback},
            state_delta={"review_feedback": review_feedback, "review_route": route},
            metadata={"forced_finalization": forced_finalization},
        )


class FinalHandler(WorkflowNodeHandler):
    def __init__(self, *, max_tokens: int, temperature: float):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def build_request(self, context) -> NodeRequest:
        payload: LoopWorkflowInput = context.workflow_input
        draft = _state_text(context.state_data, "draft")
        review_feedback = _state_text(context.state_data, "review_feedback")
        return NodeRequest.from_current_prompt(
            prompt=_final_prompt(payload, draft, review_feedback),
            sampling_params=SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        )

    def parse_output(self, context, output) -> NodeResult:
        return NodeResult(
            artifacts={"final_answer": output.text},
            state_delta={"final_answer": output.text},
        )


def build_loop_demo(
    *,
    writer_max_tokens: int = 192,
    reviewer_max_tokens: int = 160,
    final_max_tokens: int = 160,
    temperature: float = 0.0,
) -> tuple[WorkflowSpec, HandlerRegistry]:
    max_loop_firings = 4

    workflow = WorkflowSpec(
        name="writer_reviewer_loop_demo",
        nodes=(
            NodeSpec(
                id="writer",
                handler="writer",
                trigger=TriggerSpec.all_of("reviewer"),
                history=HistorySpec(system_prompt=WRITER_SYSTEM, include_ancestor_history=False),
                max_firings=max_loop_firings,
                start_on_create=True,
                description="Writer loop node. Each iteration uses explicit draft and reviewer-feedback state.",
            ),
            NodeSpec(
                id="reviewer",
                handler="reviewer",
                trigger=TriggerSpec.all_of("writer"),
                history=HistorySpec(system_prompt=REVIEWER_SYSTEM, include_ancestor_history=False),
                max_firings=max_loop_firings,
                description="Reviewer routes back to writer on revise, or forces the handoff to final on its last allowed firing.",
            ),
            NodeSpec(
                id="final",
                handler="final",
                trigger=TriggerSpec.all_of("reviewer"),
                history=HistorySpec(system_prompt=FINAL_SYSTEM, include_ancestor_history=False),
                description="Final node runs only after the reviewer emits route=approve.",
            ),
        ),
        edges=(
            EdgeSpec("writer", "reviewer"),
            EdgeSpec("reviewer", "writer", condition="revise"),
            EdgeSpec("reviewer", "final", condition="approve"),
        ),
        finish_policy=FinishPolicy.node("final"),
        limits=ExecutionLimits(
            max_total_firings=10,
            max_firings_per_node={"writer": max_loop_firings, "reviewer": max_loop_firings, "final": 1},
        ),
        description="Loop demo with writer/reviewer iterations using explicit state passing instead of ancestor history accumulation.",
    )

    handlers = HandlerRegistry(
        {
            "writer": WriterHandler(max_tokens=writer_max_tokens, temperature=temperature),
            "reviewer": ReviewerHandler(
                max_tokens=reviewer_max_tokens,
                temperature=temperature,
                max_firings=max_loop_firings,
            ),
            "final": FinalHandler(max_tokens=final_max_tokens, temperature=temperature),
        }
    )
    return workflow, handlers
