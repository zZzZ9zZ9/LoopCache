from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from loopcache.capacity import available_bytes, kv_bytes_per_token


class CapacityTests(unittest.TestCase):
    def test_unrolled_slots(self):
        ouro = SimpleNamespace(
            model_type="ouro", num_hidden_layers=24, num_key_value_heads=16, head_dim=128
        )
        self.assertEqual(kv_bytes_per_token(ouro, 4, torch.bfloat16), 786432)
        raven = SimpleNamespace(
            model_type="huginn_raven",
            n_layers_in_prelude=4,
            n_layers_in_coda=4,
            n_layers_in_recurrent_block=6,
            num_key_value_heads=8,
            hidden_size=2048,
            num_attention_heads=32,
        )
        self.assertEqual(kv_bytes_per_token(raven, 32, torch.bfloat16), 409600)
        self.assertEqual(kv_bytes_per_token(raven, 4, torch.float32), 131072)

    def test_allocator_headroom(self):
        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(60, 100)),
            patch.object(torch.cuda, "memory_reserved", return_value=35),
            patch.object(torch.cuda, "memory_allocated", return_value=25),
        ):
            for fraction, expected in [(1.0, 70), (0.5, 25), (0.2, 10)]:
                with (
                    self.subTest(fraction=fraction),
                    patch.object(
                        torch.cuda.memory,
                        "get_per_process_memory_fraction",
                        return_value=fraction,
                        create=True,
                    ),
                ):
                    self.assertEqual(available_bytes(0), expected)
            with patch.object(torch.cuda.memory, "get_per_process_memory_fraction", None, create=True):
                self.assertEqual(available_bytes(0), 70)
