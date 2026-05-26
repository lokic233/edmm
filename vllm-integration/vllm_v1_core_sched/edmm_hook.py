# SPDX-License-Identifier: Apache-2.0
"""
EDMM scheduler hook: tool-call detection, CPU workload gating,
and speculative prefill lifecycle management.

This module is imported by the scheduler only when VLLM_EDMM_ENABLE=1.
When disabled, the scheduler bypasses all EDMM code paths with zero overhead.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

EDMM_ENABLED = os.environ.get("VLLM_EDMM_ENABLE", "0") == "1"
TOOL_SENTINEL_TOKEN = "<|call_tool|>"
CPU_LOAD_THRESHOLD = 75.0


class SpeculationDecision(Enum):
    SPECULATE = auto()
    DEFER = auto()


@dataclass
class ToolCallState:
    """Tracks a single request's tool-call pause lifecycle."""

    request_id: str
    paused_at_ns: int = 0
    speculative_handles: list[int] = field(default_factory=list)
    speculation_decision: Optional[SpeculationDecision] = None
    resumed: bool = False


class EdmmSchedulerHook:
    """Stateful hook that the scheduler calls at key lifecycle points.

    Usage from scheduler:
        if EDMM_ENABLED:
            self._edmm_hook = EdmmSchedulerHook()

        # When output tokens are processed:
            if self._edmm_hook.should_pause_for_tool_call(request, new_token_ids):
                request.status = RequestStatus.WAITING_FOR_TOOL_CALL
                self._edmm_hook.on_tool_call_pause(request.request_id)

        # When checking blocked requests for promotion:
            if request.status == RequestStatus.WAITING_FOR_TOOL_CALL:
                if self._edmm_hook.is_tool_response_ready(request.request_id):
                    # promote back to WAITING
    """

    def __init__(self):
        self._active_pauses: dict[str, ToolCallState] = {}
        self._tool_responses: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cpu_load_cache: float = 0.0
        self._cpu_load_cache_time: float = 0.0

    def should_pause_for_tool_call(
        self, output_token_ids: list[int], tokenizer: Any
    ) -> bool:
        """Check if the latest output contains a tool-call sentinel."""
        if not output_token_ids:
            return False
        last_text = tokenizer.decode(
            output_token_ids[-min(10, len(output_token_ids)) :]
        )
        return TOOL_SENTINEL_TOKEN in last_text

    def on_tool_call_pause(self, request_id: str) -> SpeculationDecision:
        """Called when a request enters WAITING_FOR_TOOL_CALL.
        Evaluates CPU load and decides whether to speculate."""
        cpu_load = self._sample_cpu_load()
        decision = (
            SpeculationDecision.DEFER
            if cpu_load > CPU_LOAD_THRESHOLD
            else SpeculationDecision.SPECULATE
        )

        state = ToolCallState(
            request_id=request_id,
            paused_at_ns=time.perf_counter_ns(),
            speculation_decision=decision,
        )

        with self._lock:
            self._active_pauses[request_id] = state

        logger.info(
            "EDMM: request %s paused for tool call (CPU=%.1f%%, decision=%s)",
            request_id,
            cpu_load,
            decision.name,
        )
        return decision

    def submit_tool_response(self, request_id: str) -> None:
        """Called externally when the tool response arrives."""
        with self._lock:
            self._tool_responses[request_id] = True

    def is_tool_response_ready(self, request_id: str) -> bool:
        """Check if a tool response has arrived for this request."""
        with self._lock:
            return request_id in self._tool_responses

    def on_tool_call_resume(self, request_id: str) -> Optional[ToolCallState]:
        """Called when the request is promoted back to WAITING/RUNNING.
        Returns the pause state for metrics/logging."""
        with self._lock:
            self._tool_responses.pop(request_id, None)
            state = self._active_pauses.pop(request_id, None)

        if state:
            pause_duration_ms = (time.perf_counter_ns() - state.paused_at_ns) / 1e6
            logger.info(
                "EDMM: request %s resumed after %.1fms pause (decision was %s)",
                request_id,
                pause_duration_ms,
                (
                    state.speculation_decision.name
                    if state.speculation_decision
                    else "NONE"
                ),
            )
        return state

    def get_active_pause(self, request_id: str) -> Optional[ToolCallState]:
        with self._lock:
            return self._active_pauses.get(request_id)

    def cleanup_request(self, request_id: str) -> None:
        """Release any speculative handles held for a request.
        Must be called when a request is aborted, errors out, or finishes
        while still in WAITING_FOR_TOOL_CALL to prevent VRAM leaks."""
        with self._lock:
            self._tool_responses.pop(request_id, None)
            state = self._active_pauses.pop(request_id, None)

        if state and state.speculative_handles:
            try:
                from vllm.v1.worker.gpu.edmm_allocator import release_physical_handle

                for handle in state.speculative_handles:
                    try:
                        release_physical_handle(handle)
                    except Exception:
                        logger.warning(
                            "EDMM: failed to release speculative handle %d "
                            "for request %s",
                            handle,
                            request_id,
                        )
            except ImportError:
                pass
            logger.info(
                "EDMM: cleaned up %d speculative handles for request %s",
                len(state.speculative_handles),
                request_id,
            )

    def cleanup_all(self) -> None:
        """Release all outstanding speculative handles. Called on shutdown."""
        with self._lock:
            request_ids = list(self._active_pauses.keys())
        for rid in request_ids:
            self.cleanup_request(rid)

    def _sample_cpu_load(self) -> float:
        """Read CPU utilization from /proc/stat. Cached for 100ms."""
        now = time.monotonic()
        if now - self._cpu_load_cache_time < 0.1:
            return self._cpu_load_cache

        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            if len(parts) >= 8:
                idle = float(parts[4])
                total = sum(float(x) for x in parts[1:8])
                self._cpu_load_cache = (
                    100.0 * (1.0 - idle / total) if total > 0 else 0.0
                )
            else:
                self._cpu_load_cache = 0.0
        except Exception:
            self._cpu_load_cache = 0.0

        self._cpu_load_cache_time = now
        return self._cpu_load_cache
