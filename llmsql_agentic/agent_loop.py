"""Custom tool-only agent loop for LLMSQL on top of verl."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from llmsql_agentic.tool_env import LLMSQLToolEnv


@dataclass
class TraceTurn:
    assistant_text: str
    action_type: str
    action_sql: str | None
    observation_text: str | None


@register("llmsql_tool_agent")
class LLMSQLToolAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        db_path: str = "/root/shared-nvme/rlvr/datasets/llmsql-2.0/sqlite_tables.db",
        max_tool_turns: int = 2,
        preview_rows: int = 5,
        preview_max_chars: int = 2000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.max_tool_turns = max_tool_turns
        self.preview_rows = preview_rows
        self.preview_max_chars = preview_max_chars
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        table_id = kwargs["extra_info"]["table_id"]

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        env = LLMSQLToolEnv(
            db_path=self.db_path,
            preview_rows=self.preview_rows,
            preview_max_chars=self.preview_max_chars,
        )

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        trace: list[dict[str, Any]] = []
        tool_turns = 0
        forced_final_turn = False
        assistant_turns = 0
        final_sql_found = False
        routed_experts = None

        while True:
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            assistant_turns += 1
            turn_token_ids = output.token_ids
            turn_logprobs = output.log_probs or []
            turn_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(turn_token_ids, skip_special_tokens=True),
            )

            response_ids.extend(turn_token_ids)
            response_mask.extend([1] * len(turn_token_ids))
            if turn_logprobs:
                response_logprobs.extend(turn_logprobs)
            prompt_ids += turn_token_ids
            if output.routed_experts is not None:
                routed_experts = output.routed_experts

            force_final_after_obs = False
            if forced_final_turn:
                parsed = env.parse_action(turn_text)
                if parsed.action_type == "final_sql":
                    final_sql_found = True
                trace.append(
                    {
                        "assistant_text": turn_text,
                        "action_type": parsed.action_type,
                        "action_sql": parsed.sql,
                        "observation_text": None,
                    }
                )
                break

            step_result = env.step(
                action_text=turn_text,
                table_id=table_id,
                final_turn=False,
            )
            trace.append(
                {
                    "assistant_text": turn_text,
                    "action_type": step_result.action_type,
                    "action_sql": step_result.sql,
                    "observation_text": step_result.observation_text or None,
                }
            )

            if step_result.is_final:
                final_sql_found = True
                break

            if step_result.action_type == "sql":
                tool_turns += 1
                if tool_turns >= self.max_tool_turns:
                    force_final_after_obs = True

            if assistant_turns >= self.max_tool_turns + 1:
                break

            observation_text = step_result.observation_text
            if not observation_text:
                break

            if force_final_after_obs:
                observation_text = (
                    f"{observation_text}\n\nThis is your final turn. Output <final_sql> ... </final_sql> now."
                )
                forced_final_turn = True

            observation_message = {"role": "user", "content": observation_text}
            messages.append({"role": "assistant", "content": turn_text})
            messages.append(observation_message)

            observation_ids = await self.apply_chat_template(
                [observation_message],
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                remove_system_prompt=True,
            )
            if len(response_mask) + len(observation_ids) >= self.response_length:
                break

            prompt_ids += observation_ids
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if response_logprobs:
                response_logprobs.extend([0.0] * len(observation_ids))

            if len(response_mask) >= self.response_length:
                break

        extra_fields = {
            "agent_trace": trace,
            "tool_turns": tool_turns,
            "forced_final_turn": forced_final_turn,
            "final_sql_found": final_sql_found,
            "turn_scores": [],
            "tool_rewards": [],
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids[: len(prompt_ids) - len(response_ids)],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=len(trace) + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
