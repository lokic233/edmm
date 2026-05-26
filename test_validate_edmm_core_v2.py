#!/usr/bin/env python3
import importlib.util
import inspect
import os
import tempfile
import unittest

import torch


SCRIPT_PATH = "/home/dengcchi/validate_edmm_core_v2.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("validate_edmm_core_v2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    def encode(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
    ):
        del add_special_tokens
        ids = [(ord(char) % 251) + 1 for char in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class ValidateEDMMCoreV2Tests(unittest.TestCase):
    def setUp(self):
        self.bench = load_benchmark_module()
        self.original_base_tokens = self.bench.BASE_CONTEXT_TOKENS
        self.original_suffix_tokens = self.bench.SUFFIX_TOKENS
        self.original_fallbacks = self.bench.LOCAL_MODEL_FALLBACKS

    def tearDown(self):
        self.bench.BASE_CONTEXT_TOKENS = self.original_base_tokens
        self.bench.SUFFIX_TOKENS = self.original_suffix_tokens
        self.bench.LOCAL_MODEL_FALLBACKS = self.original_fallbacks
        os.environ.pop("EDMM_MODEL_NAME", None)

    def test_workload_builder_enforces_exact_base_and_suffix_block_lengths(self):
        self.bench.BASE_CONTEXT_TOKENS = 64
        self.bench.SUFFIX_TOKENS = 16

        inputs = self.bench.build_benchmark_inputs(CharTokenizer(), "cpu")

        self.assertEqual(inputs["base_ids"].shape[1], 64)
        self.assertGreater(inputs["suffix_ids"].shape[1], 16)
        self.assertTrue(inputs["base_ids"].is_contiguous())
        self.assertTrue(inputs["suffix_ids"].is_contiguous())

    def test_group_b_and_group_c_use_same_downstream_suffix_tensor(self):
        self.bench.BASE_CONTEXT_TOKENS = 64
        self.bench.SUFFIX_TOKENS = 16
        inputs = self.bench.build_benchmark_inputs(CharTokenizer(), "cpu")
        dynamic_ids = self.bench.tokenize_to_device(
            CharTokenizer(),
            "\nERROR_LOG_ID_test TIMESTAMP_1 SESSION_test TRACE_ID_test",
            "cpu",
        )

        full_recompute_ids = self.bench.build_full_recompute_ids(
            inputs["base_ids"],
            dynamic_ids,
            inputs["suffix_ids"],
        )
        speculative_prefix_ids = self.bench.build_speculative_prefix_ids(
            inputs["base_ids"],
            dynamic_ids,
        )

        self.assertEqual(
            full_recompute_ids.shape[1],
            inputs["base_ids"].shape[1]
            + dynamic_ids.shape[1]
            + inputs["suffix_ids"].shape[1],
        )
        self.assertEqual(
            speculative_prefix_ids.shape[1],
            inputs["base_ids"].shape[1] + dynamic_ids.shape[1],
        )
        self.assertTrue(
            torch.equal(
                full_recompute_ids[:, -inputs["suffix_ids"].shape[1] :],
                inputs["suffix_ids"],
            )
        )

    def test_clone_kv_cache_copies_tensors_without_aliasing(self):
        key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
        value = torch.arange(8, 16, dtype=torch.float32).reshape(1, 1, 2, 4)
        original = ((key, value),)

        cloned = self.bench.clone_kv_cache(original)

        self.assertTrue(torch.equal(cloned[0][0], key))
        self.assertTrue(torch.equal(cloned[0][1], value))
        self.assertNotEqual(cloned[0][0].data_ptr(), key.data_ptr())
        self.assertNotEqual(cloned[0][1].data_ptr(), value.data_ptr())

        key.add_(100)
        self.assertFalse(torch.equal(cloned[0][0], key))

    def test_no_deepcopy_usage_remains_in_hot_path(self):
        source = inspect.getsource(self.bench)

        self.assertNotIn("copy.deepcopy", source)
        self.assertNotIn("\nimport copy", source)

    def test_model_resolution_prefers_override_then_local_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bench.LOCAL_MODEL_FALLBACKS = (tmpdir,)
            os.environ["EDMM_MODEL_NAME"] = "/explicit/model"
            self.assertEqual(self.bench.resolve_model_name(), "/explicit/model")

            os.environ.pop("EDMM_MODEL_NAME")
            self.assertEqual(self.bench.resolve_model_name(), tmpdir)


if __name__ == "__main__":
    unittest.main()
