"""Focused invariants for the contextual token readouts (TR-C)."""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from contextual_readout_training import (  # noqa: E402
    aligned_windows,
    all_position_loss,
)
from models import (  # noqa: E402
    DescriptionContextualTokenReadout,
    DescriptionSubstratePredictor,
    PixelContextualTokenReadout,
    PixelSubstratePredictor,
)


class ContextualReadoutInvariantTests(unittest.TestCase):

    def _assert_future_independent(self, model, trace, split):
        altered = trace.detach().clone()
        altered[:, split:] = torch.randn_like(altered[:, split:]) * 10
        for training in (False, True):
            model.train(training)
            with torch.no_grad():
                before = model(trace)[:, :split]
                after = model(altered)[:, :split]
            torch.testing.assert_close(before, after, rtol=0, atol=1e-6)

    def _assert_future_gradient_zero(self, model, trace, position):
        model.train()
        differentiable = trace.detach().clone().requires_grad_(True)
        model(differentiable)[:, position].sum().backward()
        future_gradient = differentiable.grad[:, position + 1:]
        self.assertEqual(torch.count_nonzero(future_gradient).item(), 0)

    def test_pixel_readout_is_causal_in_train_and_eval(self):
        torch.manual_seed(1)
        model = PixelContextualTokenReadout(
            img_size=16,
            n_tokens=19,
            context_len=6,
            d_model=16,
            n_heads=4,
            n_layers=2,
            ff_dim=32,
            pool_grid=4,
        )
        trace = torch.rand(2, 6, 3, 16, 16)
        self._assert_future_independent(model, trace, split=3)
        self._assert_future_gradient_zero(model, trace, position=2)
        self.assertEqual(model(trace).shape, (2, 6, 19))

    def test_description_readout_is_causal_in_train_and_eval(self):
        torch.manual_seed(2)
        model = DescriptionContextualTokenReadout(
            embed_dim=12,
            desc_len=5,
            n_tokens=19,
            context_len=6,
            d_model=12,
            temporal_heads=3,
            temporal_layers=2,
            temporal_ff_dim=24,
            pool_bins=3,
        )
        trace = torch.rand(2, 6, 5, 12)
        self._assert_future_independent(model, trace, split=3)
        self._assert_future_gradient_zero(model, trace, position=2)
        self.assertEqual(model(trace).shape, (2, 6, 19))

    def test_readouts_reject_raw_token_id_shapes(self):
        pixel = PixelContextualTokenReadout(
            img_size=16,
            n_tokens=19,
            context_len=6,
            d_model=16,
            n_heads=4,
            n_layers=1,
            ff_dim=32,
            pool_grid=4,
        )
        description = DescriptionContextualTokenReadout(
            embed_dim=12,
            desc_len=5,
            n_tokens=19,
            context_len=6,
            d_model=12,
            temporal_heads=3,
            temporal_layers=1,
            temporal_ff_dim=24,
            pool_bins=3,
        )
        token_ids = torch.ones(2, 6, dtype=torch.long)
        with self.assertRaises(ValueError):
            pixel(token_ids)
        with self.assertRaises(ValueError):
            description(token_ids)

    def test_alignment_uses_each_sp_state_at_the_same_target_index(self):
        stream = torch.arange(10)
        inputs, targets = aligned_windows(
            stream, starts=[0, 5], context_len=4, device="cpu")
        torch.testing.assert_close(
            inputs, torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]]))
        torch.testing.assert_close(
            targets, torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]]))
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
        with self.assertRaises(ValueError):
            aligned_windows(
                stream, starts=[6], context_len=4, device="cpu")

    def test_training_loss_supervises_every_aligned_position(self):
        torch.manual_seed(3)
        logits = torch.randn(2, 4, 7, requires_grad=True)
        targets = torch.randint(0, 7, (2, 4))
        actual = all_position_loss(logits, targets, reduction="sum")
        expected = F.cross_entropy(
            logits.reshape(-1, 7), targets.reshape(-1), reduction="sum")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.count_nonzero(logits.grad[:, :-1]).item() > 0)
        self.assertTrue(torch.count_nonzero(logits.grad[:, -1:]).item() > 0)

    def test_full_sp_states_equal_the_corresponding_prefix_states(self):
        torch.manual_seed(4)
        ids = torch.randint(0, 17, (2, 5))
        predictors = (
            PixelSubstratePredictor(
                n_tokens=17,
                img_size=4,
                img_ch=3,
                context_len=5,
                txt_layers=1,
                txt_d_model=8,
                txt_heads=2,
            ),
            DescriptionSubstratePredictor(
                n_tokens=17,
                desc_len=3,
                embed_dim=4,
                context_len=5,
                txt_layers=1,
                txt_d_model=8,
                txt_heads=2,
            ),
        )
        for predictor in predictors:
            predictor.eval()
            full_trace = predictor(ids)
            for position in range(ids.shape[1]):
                prefix_state = predictor.forward_last(ids[:, :position + 1])
                torch.testing.assert_close(
                    full_trace[:, position],
                    prefix_state,
                    rtol=0,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
