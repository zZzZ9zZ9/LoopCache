from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from loopcache.placement import RequestPlacement


class Model:
    def __init__(self, devices):
        self.devices = tuple(devices)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        config = SimpleNamespace(
            model_type="ouro", num_hidden_layers=1, num_key_value_heads=1, head_dim=1
        )
        self.devices = tuple(map(torch.device, ("cuda:0", "cuda:1")))
        self.free = dict.fromkeys(self.devices, 1024)
        self.weight_errors = {}
        self.decode_errors = {}
        self.loaded = []
        self.executed = []
        self.probed = []
        self.live_singles_on_parallel_load = []
        stack.enter_context(patch("loopcache.placement.AutoConfig.from_pretrained", return_value=config))
        stack.enter_context(
            patch("loopcache.placement.AutoTokenizer.from_pretrained", return_value=object())
        )
        self.set_device = stack.enter_context(patch("torch.cuda.set_device"))
        stack.enter_context(patch("torch.cuda.empty_cache"))
        stack.enter_context(patch("loopcache.placement.available_bytes", self.available))
        stack.enter_context(patch("loopcache.batching.model.BatchModel", self.load))
        stack.enter_context(patch("loopcache.model.LoopCacheModel", self.load))
        stack.enter_context(patch("loopcache.runtime.generate", self.single))
        stack.enter_context(patch("loopcache.batching.generate_batch", self.batch))

    def runner(self, size=1):
        placement = RequestPlacement("ouro-1.4b", devices=self.devices, batch_size=size)
        self.addCleanup(placement.close)
        return placement

    def available(self, device):
        self.probed.append(device)
        return self.free[device]

    def load(self, name, checkpoint, *, devices=None, device=None, **kwargs):
        model = Model(devices or (device,))
        if len(model.devices) > 1:
            self.live_singles_on_parallel_load = [
                ds for ds, ref in self.loaded if len(ds) == 1 and ref() is not None
            ]
        self.loaded.append((model.devices, weakref.ref(model)))
        if model.devices in self.weight_errors:
            raise self.weight_errors[model.devices]()
        return model

    def batch(self, model, rows, budget, **settings):
        self.executed.append((model.devices, [row[0] for row in rows]))
        if model.devices in self.decode_errors:
            raise self.decode_errors[model.devices]()
        return dict(
            token_ids=[[row[0]] * budget for row in rows],
            decode_tokens=len(rows) * (budget - 1),
            decode_seconds=1.0,
            prefill_seconds=1.0,
            graph_setup_seconds=1.0,
            cuda_graph=True,
            method=1,
        )

    def single(self, model, row, *, max_new_tokens, **settings):
        result = self.batch(model, [row], max_new_tokens, **settings)
        result.update(token_ids=result["token_ids"][0], graph_replays=1)
        return result

    def test_primary_success_leaves_other_devices_untouched(self):
        result = self.runner().run([[1]], 2)
        primary = self.devices[0]
        self.assertEqual(self.executed, [((primary,), [1])])
        self.assertEqual(set(self.probed), {primary})
        self.assertTrue(all(call.args == (primary,) for call in self.set_device.call_args_list))
        self.assertEqual(result["placement"], "single")

    def test_secondary_can_serve_single_request(self):
        first, second = self.devices

        def free_after_weights(device):
            loaded_here = any(devices == (device,) and ref() is not None for devices, ref in self.loaded)
            return 47 if device == first and loaded_here else 1024

        with patch("loopcache.placement.available_bytes", free_after_weights):
            result = self.runner().run([[7]], 2)
        self.assertEqual(self.executed, [((second,), [7])])
        self.assertEqual(result["placement"], "single")

    def test_batch_splits_without_reordering(self):
        self.free = dict.fromkeys(self.devices, 96)
        rows = [[3], [1, 4, 5], [2, 8]]
        result = self.runner(size=3).run(rows, 2)
        self.assertEqual(result["token_ids"], [[3, 3], [1, 1], [2, 2]])
        self.assertEqual([b["placement"] for b in result["batches"]], ["single"] * 3)

    def test_parallel_follows_all_single_device_decode_ooms(self):
        self.decode_errors = {(dev,): torch.cuda.OutOfMemoryError for dev in self.devices}
        result = self.runner().run([[5]], 2)
        self.assertEqual(
            [devs for devs, _ in self.executed], [(self.devices[0],), (self.devices[1],), self.devices]
        )
        self.assertFalse(self.live_singles_on_parallel_load)
        self.assertTrue(all(ref() is None for devices, ref in self.loaded if len(devices) == 1))
        self.assertEqual((result["placement"], result["gpu_count"]), ("layer", 2))

    def test_weight_oom_does_not_fall_back_to_parallel(self):
        self.weight_errors = {(dev,): torch.cuda.OutOfMemoryError for dev in self.devices}
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.runner().run([[5]], 2)
        self.assertFalse(self.executed)

    def test_weight_oom_with_other_device_full(self):
        first, second = self.devices
        self.weight_errors[(first,)] = torch.cuda.OutOfMemoryError
        self.free[second] = 47
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.runner().run([[5]], 2)
        self.assertFalse(self.executed)

    def test_other_decode_errors_propagate(self):
        self.decode_errors[(self.devices[0],)] = ValueError
        with self.assertRaises(ValueError):
            self.runner().run([[5]], 2)
        self.assertEqual(self.executed, [((self.devices[0],), [5])])
