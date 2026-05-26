"""
EDMM Phase 3 validation: Scheduler hook lifecycle.

Tests the tool-call pause/resume state machine and CPU workload gating
without requiring a GPU or a running model.
"""

import os
import time

import pytest

os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm.v1.core.sched.edmm_hook import (
    EdmmSchedulerHook,
    SpeculationDecision,
    ToolCallState,
)


class TestEdmmSchedulerHook:

    def setup_method(self):
        self.hook = EdmmSchedulerHook()

    def test_pause_creates_state(self):
        decision = self.hook.on_tool_call_pause("req-1")
        assert decision in (SpeculationDecision.SPECULATE, SpeculationDecision.DEFER)
        state = self.hook.get_active_pause("req-1")
        assert state is not None
        assert state.request_id == "req-1"
        assert state.paused_at_ns > 0

    def test_tool_response_not_ready_before_submit(self):
        self.hook.on_tool_call_pause("req-2")
        assert not self.hook.is_tool_response_ready("req-2")

    def test_tool_response_ready_after_submit(self):
        self.hook.on_tool_call_pause("req-3")
        self.hook.submit_tool_response("req-3")
        assert self.hook.is_tool_response_ready("req-3")

    def test_resume_clears_state(self):
        self.hook.on_tool_call_pause("req-4")
        self.hook.submit_tool_response("req-4")
        state = self.hook.on_tool_call_resume("req-4")
        assert state is not None
        assert state.request_id == "req-4"
        # After resume, state is cleared
        assert self.hook.get_active_pause("req-4") is None
        assert not self.hook.is_tool_response_ready("req-4")

    def test_resume_without_pause_returns_none(self):
        state = self.hook.on_tool_call_resume("req-never-paused")
        assert state is None

    def test_cpu_high_load_defers(self):
        # Mock CPU load to be above threshold
        self.hook._sample_cpu_load = lambda: 90.0
        decision = self.hook.on_tool_call_pause("req-5")
        assert decision == SpeculationDecision.DEFER
        state = self.hook.get_active_pause("req-5")
        assert state.speculation_decision == SpeculationDecision.DEFER

    def test_cpu_low_load_speculates(self):
        self.hook._sample_cpu_load = lambda: 10.0
        decision = self.hook.on_tool_call_pause("req-6")
        assert decision == SpeculationDecision.SPECULATE
        state = self.hook.get_active_pause("req-6")
        assert state.speculation_decision == SpeculationDecision.SPECULATE

    def test_multiple_concurrent_pauses(self):
        self.hook._sample_cpu_load = lambda: 5.0
        self.hook.on_tool_call_pause("req-a")
        self.hook.on_tool_call_pause("req-b")
        self.hook.on_tool_call_pause("req-c")

        assert self.hook.get_active_pause("req-a") is not None
        assert self.hook.get_active_pause("req-b") is not None
        assert self.hook.get_active_pause("req-c") is not None

        # Submit response for b only
        self.hook.submit_tool_response("req-b")
        assert not self.hook.is_tool_response_ready("req-a")
        assert self.hook.is_tool_response_ready("req-b")
        assert not self.hook.is_tool_response_ready("req-c")

        # Resume b
        self.hook.on_tool_call_resume("req-b")
        assert self.hook.get_active_pause("req-b") is None
        # a and c still paused
        assert self.hook.get_active_pause("req-a") is not None
        assert self.hook.get_active_pause("req-c") is not None

    def test_pause_duration_tracking(self):
        self.hook.on_tool_call_pause("req-7")
        time.sleep(0.01)
        self.hook.submit_tool_response("req-7")
        state = self.hook.on_tool_call_resume("req-7")
        assert state is not None
        # Pause should have been at least 10ms
        pause_ns = time.perf_counter_ns() - state.paused_at_ns
        assert pause_ns > 5_000_000  # > 5ms

    def test_cpu_load_reads_proc_stat(self):
        """Verify real /proc/stat reading works (not mocked)."""
        hook = EdmmSchedulerHook()
        load = hook._sample_cpu_load()
        assert 0.0 <= load <= 100.0

    def test_request_status_enum_has_tool_call(self):
        """Verify WAITING_FOR_TOOL_CALL exists in the source file."""
        import ast

        src_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "vllm", "v1", "request.py"
        )
        with open(src_path) as f:
            tree = ast.parse(f.read())
        enum_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "RequestStatus":
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                enum_names.append(target.id)
        assert "WAITING_FOR_TOOL_CALL" in enum_names
        assert enum_names.index("WAITING_FOR_TOOL_CALL") < enum_names.index("RUNNING")

    def test_cleanup_request_clears_state(self):
        self.hook.on_tool_call_pause("req-cleanup")
        self.hook.submit_tool_response("req-cleanup")
        self.hook.cleanup_request("req-cleanup")
        assert self.hook.get_active_pause("req-cleanup") is None
        assert not self.hook.is_tool_response_ready("req-cleanup")

    def test_cleanup_nonexistent_request_is_safe(self):
        self.hook.cleanup_request("req-never-existed")

    def test_cleanup_all_clears_everything(self):
        self.hook.on_tool_call_pause("req-x")
        self.hook.on_tool_call_pause("req-y")
        self.hook.cleanup_all()
        assert self.hook.get_active_pause("req-x") is None
        assert self.hook.get_active_pause("req-y") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
