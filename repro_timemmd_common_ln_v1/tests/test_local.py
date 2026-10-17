"""CPU-only unit tests for the EXP-028 LN package (no real MAE weights;
VisionTS is constructed with load_ckpt=False, random init).

Covers: LN inventory 84/55,808 + freeze pattern; entry equivalence
(grad vs inference) on CPU; microbatch accumulation identity in f64
(1e-12) and in f32 via logical_batch_step; exactly one optimizer step per
logical batch with grads consumed between batches; freeze forensics paths;
checkpoint roundtrip + tamper rejection + cross-group binding rejection;
fingerprint stability/discrimination; run_train artifact checks; package
import isolation (only preflight may import the EXP-027 package).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_ln_v1 import protocol as P
from repro_timemmd_common_ln_v1.adapter import build_visionts, configure, \
    forward_batch, forward_batch_grad, freeze_to_ln, ln_param_names
from repro_timemmd_common_ln_v1.common import buffer_digest, \
    model_state_digest, named_tensor_sha, pred_close
from repro_timemmd_common_ln_v1.ln_train import freeze_check, \
    group_fingerprint, logical_batch_step, npz_key, \
    oneshot_logical_batch_grads, restore_ln_ckpt_strict, save_ln_ckpt, \
    train_config
from repro_timemmd_common_ln_v1.run_train import group_artifacts_ok


def _fresh_vts(L=96, H=12, p=1):
    torch.manual_seed(P.SEED)
    m = build_visionts(load_ckpt=False)
    configure(m, L, H, p)
    freeze_to_ln(m)
    return m


class DummyTS(torch.nn.Module):
    """Tiny stand-in with the VisionTS forward signature."""

    def __init__(self, dim=8, H=3):
        super().__init__()
        self.pred_len = H
        self.ln1 = torch.nn.LayerNorm(dim)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.lin2 = torch.nn.Linear(dim, H)

    def forward(self, x, export_image=False, fp64=False):
        y = self.lin2(torch.tanh(self.ln2(self.lin1(self.ln1(
            x.squeeze(-1))))))
        return y.unsqueeze(-1)


def _adam(m):
    return torch.optim.Adam(
        [p for p in m.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


class TestInventory(unittest.TestCase):
    def test_ln_inventory_84_55808(self):
        m = _fresh_vts()
        names = ln_param_names(m)
        ps = dict(m.named_parameters())
        self.assertEqual(len(names), P.EXPECTED_LN_TENSORS)
        self.assertEqual(sum(ps[n].numel() for n in names),
                         P.EXPECTED_LN_SCALARS)
        # inventory is configuration-independent
        configure(m, 1056, 48, 4)
        self.assertEqual(ln_param_names(m), names)

    def test_freeze_pattern(self):
        m = _fresh_vts()
        names = set(ln_param_names(m))
        for n, prm in m.named_parameters():
            self.assertEqual(prm.requires_grad, n in names,
                             "requires_grad wrong on %s" % n)
        self.assertEqual(len(names), 84)

    def test_forward_entries_close_cpu(self):
        m = _fresh_vts(96, 12, 1)
        torch.manual_seed(0)
        x = torch.randn(2, 96, 1)
        y_inf = forward_batch(m, x).numpy()
        y_grad = forward_batch_grad(m, x).detach().numpy()
        close, _, viol = pred_close(y_grad, y_inf, 1e-6, 1e-5)
        self.assertTrue(close, "entry mismatch viol=%.2e" % viol)
        # grad path actually flows to LN tensors
        forward_batch_grad(m, x).pow(2).sum().backward()
        n_grad = sum(1 for p in m.parameters()
                     if p.requires_grad and p.grad is not None
                     and torch.isfinite(p.grad).all())
        self.assertEqual(n_grad, 84)


class TestAccumulation(unittest.TestCase):
    def _f64_math(self, n, mb, H=3, dim=12):
        torch.manual_seed(1)
        m = DummyTS(dim, H).double()
        X = torch.randn(n, dim, 1).double()
        Y = torch.randn(n, H, 1).double()
        denom = float(n * H)
        # accumulated
        m.zero_grad()
        sse_tot = 0.0
        n_mb = 0
        for lo in range(0, n, mb):
            xb, yb = X[lo:lo + mb], Y[lo:lo + mb]
            out = m(xb)
            sse = ((out - yb) ** 2).sum()
            (sse / denom).backward()
            sse_tot += float(sse.item())
            n_mb += 1
        g_acc = {n_: p.grad.clone() for n_, p in m.named_parameters()}
        # one-shot
        m.zero_grad()
        out = m(X)
        loss = ((out - Y) ** 2).mean()
        loss.backward()
        g_one = {n_: p.grad.clone() for n_, p in m.named_parameters()}
        for k in g_acc:
            self.assertTrue(torch.allclose(g_acc[k], g_one[k],
                                           rtol=0, atol=1e-12),
                            "grad mismatch %s (n=%d mb=%d)" % (k, n, mb))
        self.assertAlmostEqual(sse_tot / denom, float(loss.item()),
                               places=12)
        self.assertEqual(n_mb, -(-n // mb))

    def test_f64_identity_ragged_7_3(self):
        self._f64_math(7, 3)

    def test_f64_identity_exact_boundary(self):
        self._f64_math(8, 8)

    def test_f64_identity_4x8(self):
        self._f64_math(32, 8)

    def test_logical_batch_step_mechanics(self):
        torch.manual_seed(2)
        m = DummyTS(12, 3)
        n = 10
        X = np.random.RandomState(0).randn(n, 12).astype(np.float32)
        Y = np.random.RandomState(1).randn(n, 3, 1).astype(np.float32)
        opt = _adam(m)
        steps = [0]
        orig = opt.step

        def counting(*a, **k):
            steps[0] += 1
            return orig(*a, **k)
        opt.step = counting
        loss, info = logical_batch_step(m, X, Y, np.arange(n),
                                        "cpu", opt, 4, apply_step=False)
        self.assertEqual(steps[0], 0)
        self.assertEqual(info["n_microbatches"], 3)  # 4+4+2 tail
        with torch.no_grad():
            out = m(torch.from_numpy(X))
            ref = float(((out - torch.from_numpy(Y)) ** 2).mean())
        # reported loss equals the full-element MSE of this batch
        self.assertAlmostEqual(loss, ref, places=5)
        m.zero_grad()
        # exactly one optimizer.step per call
        loss, info = logical_batch_step(m, X, Y, np.arange(n),
                                        "cpu", opt, 4)
        self.assertEqual(steps[0], 1)
        self.assertAlmostEqual(loss, ref, places=5)
        # grads left by a no-step accumulation are those of THIS batch only
        opt.zero_grad()
        logical_batch_step(m, X, Y, np.arange(4, n), "cpu", opt, 32,
                           apply_step=False)
        g_lb = {n_: prm.grad.clone() for n_, prm in m.named_parameters()}
        _, g_step = oneshot_logical_batch_grads(
            m, X, Y, np.arange(4, n), "cpu")
        for n_ in g_lb:
            self.assertTrue(torch.allclose(g_lb[n_], g_step[n_],
                                           rtol=1e-5, atol=1e-7))
        m.zero_grad()

    def test_oneshot_matches_manual(self):
        torch.manual_seed(3)
        m = DummyTS(12, 3)
        X = np.random.RandomState(0).randn(8, 12).astype(np.float32)
        Y = np.random.RandomState(1).randn(8, 3, 1).astype(np.float32)
        loss, grads = oneshot_logical_batch_grads(m, X, Y,
                                                  np.arange(8), "cpu")
        with torch.no_grad():
            out = m(torch.from_numpy(X))
            ref = float(((out - torch.from_numpy(Y)) ** 2).mean())
        self.assertAlmostEqual(loss, ref, places=6)
        self.assertEqual(len(grads), sum(1 for p in m.parameters()
                                         if p.requires_grad))


def _fake_model():
    root = torch.nn.Module()
    vm = torch.nn.Module()
    root.vision_model = vm
    vm.blocks = torch.nn.Module()
    for i in range(2):
        b = torch.nn.Module()
        vm.blocks.__setattr__("b%d" % i, b)
        b.norm1 = torch.nn.LayerNorm(4)
        b.norm2 = torch.nn.LayerNorm(4)
    vm.norm = torch.nn.LayerNorm(4)
    vm.decoder_blocks = torch.nn.Module()
    b = torch.nn.Module()
    vm.decoder_blocks.__setattr__("b0", b)
    b.norm1 = torch.nn.LayerNorm(4)
    vm.decoder_norm = torch.nn.LayerNorm(4)
    root.head = torch.nn.Linear(4, 2)
    root.register_buffer("scratch", torch.zeros(3))
    return root


class TestFreezeCheck(unittest.TestCase):
    def _bump(self, m, which):
        with torch.no_grad():
            if "enc" in which:
                m.vision_model.blocks.b0.norm1.weight.add_(0.1)
            if "dec" in which:
                m.vision_model.decoder_norm.bias.add_(0.1)
            if "nonln" in which:
                m.head.weight.add_(0.1)
            if "buf" in which:
                m.scratch.add_(1.0)

    def test_enc_and_dec_moved_ok(self):
        m = _fake_model()
        before = named_tensor_sha(m)
        buf = buffer_digest(m)
        # no change at all -> encoder/decoder groups did not move
        chk = freeze_check(m, before, buf)
        self.assertFalse(chk["ok"])
        self._bump(m, "enc+dec")
        chk = freeze_check(m, before, buf)
        self.assertTrue(chk["ok"])
        self.assertEqual(chk["n_moved"], 2)
        self.assertEqual(len(chk["unchanged_ln"]), 12)

    def test_only_encoder_moved_fails(self):
        m = _fake_model()
        before = named_tensor_sha(m)
        buf = buffer_digest(m)
        self._bump(m, "enc")
        chk = freeze_check(m, before, buf)
        self.assertFalse(chk["ok"])
        self.assertEqual(len(chk["decoder_moved"]), 0)

    def test_non_ln_moved_fails(self):
        m = _fake_model()
        before = named_tensor_sha(m)
        buf = buffer_digest(m)
        self._bump(m, "enc+dec+nonln")
        chk = freeze_check(m, before, buf)
        self.assertFalse(chk["ok"])
        self.assertEqual(chk["non_ln_moved"], ["head.weight"])

    def test_buffer_change_fails(self):
        m = _fake_model()
        before = named_tensor_sha(m)
        buf0 = buffer_digest(m)
        self._bump(m, "enc+dec+buf")
        chk = freeze_check(m, before, buf0)
        self.assertFalse(chk["ok"])
        self.assertFalse(chk["buffer_unchanged"])


class TestCkpt(unittest.TestCase):
    def test_roundtrip_tamper_binding(self):
        tmp = Path(tempfile.mkdtemp())
        m1 = _fresh_vts()
        names = ln_param_names(m1)
        digest_before = model_state_digest(m1)
        with torch.no_grad():
            for n in names[:5]:
                dict(m1.named_parameters())[n].add_(0.01)
        digest_after = model_state_digest(m1)
        self.assertNotEqual(digest_before, digest_after)
        meta_base = {
            "gid": "Health__H12__t", "domain": "Health", "h": 12, "p": 1,
            "L": 96,
            "protocol_name": P.PROTOCOL_NAME,
            "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
            "splits_json_sha256": P.SPLITS_JSON_SHA256,
            "scaler_json_sha256": P.SCALER_JSON_SHA256,
            "weights_sha256": "0" * 64,
            "train_config": train_config(32),
            "steps": 1, "n_train_windows": 256,
            "digest_before": digest_before, "digest_after": digest_after,
        }
        save_ln_ckpt(m1, names, tmp / "ck.npz", tmp / "meta.json",
                     meta_base)
        pinfo = {"weights_sha256": "0" * 64}

        m2 = _fresh_vts()
        restore_ln_ckpt_strict(m2, tmp / "ck.npz", tmp / "meta.json",
                               pinfo, "Health", 12)
        self.assertEqual(model_state_digest(m2), digest_after)
        # freeze pattern preserved after restore
        for n, prm in m2.named_parameters():
            self.assertEqual(prm.requires_grad, n in set(names))

        # tamper: rewrite a tensor, keep meta -> file sha gate trips
        z = dict(np.load(str(tmp / "ck.npz")))
        z[npz_key(names[0])] = (z[npz_key(names[0])] + 1.0) \
            .astype(np.float32)
        np.savez(str(tmp / "ck_tampered.npz"), **z)
        with self.assertRaises(SystemExit):
            restore_ln_ckpt_strict(m2, tmp / "ck_tampered.npz",
                                   tmp / "meta.json", pinfo,
                                   "Health", 12)

        # binding: same file, wrong domain -> SystemExit
        with self.assertRaises(SystemExit):
            restore_ln_ckpt_strict(m2, tmp / "ck.npz", tmp / "meta.json",
                                   pinfo, "Climate", 12)
        # wrong base weights pin -> SystemExit
        with self.assertRaises(SystemExit):
            restore_ln_ckpt_strict(m2, tmp / "ck.npz", tmp / "meta.json",
                                   {"weights_sha256": "f" * 64},
                                   "Health", 12)


class TestFingerprintAndArtifacts(unittest.TestCase):
    def _pinfo(self, w="a" * 64, n_train=5000):
        return {"weights_sha256": w,
                "splits": {"Health": {"window_counts":
                                      {"12": {"train": n_train}}},
                           "Climate": {"window_counts":
                                       {"3": {"train": n_train}}}}}

    def test_fingerprint_stable_and_discriminating(self):
        f1 = group_fingerprint(self._pinfo(), "Health", 12,
                               train_config(32))
        f2 = group_fingerprint(self._pinfo(), "Health", 12,
                               train_config(32))
        self.assertEqual(f1, f2)
        self.assertNotEqual(f1, group_fingerprint(
            self._pinfo(w="b" * 64), "Health", 12, train_config(32)))
        self.assertNotEqual(f1, group_fingerprint(
            self._pinfo(), "Health", 12, train_config(16)))
        self.assertNotEqual(f1, group_fingerprint(
            self._pinfo(n_train=5001), "Health", 12, train_config(32)))
        self.assertNotEqual(f1, group_fingerprint(
            self._pinfo(), "Climate", 3, train_config(32)))

    def test_group_artifacts_ok(self):
        tmp = Path(tempfile.mkdtemp())
        m = _fresh_vts()
        names = ln_param_names(m)
        meta_base = {
            "gid": "Security_H6", "domain": "Security", "h": 6, "p": 6,
            "L": 220,
            "protocol_name": P.PROTOCOL_NAME,
            "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
            "splits_json_sha256": P.SPLITS_JSON_SHA256,
            "scaler_json_sha256": P.SCALER_JSON_SHA256,
            "weights_sha256": "0" * 64,
            "train_config": train_config(32),
            "steps": P.expected_steps(12), "n_train_windows": 12,
        }
        save_ln_ckpt(m, names, tmp / P.ckpt_npz_name("Security", 6),
                     tmp / P.ckpt_meta_name("Security", 6), meta_base)
        np.savez(str(tmp / P.val_npz_name("Security", 6)),
                 pred=np.zeros((3, 6, 1), np.float32),
                 truth=np.zeros((3, 6, 1), np.float32),
                 window_index=np.arange(3))
        self.assertTrue(group_artifacts_ok(tmp, "Security", 6, 3))
        # wrong val count
        self.assertFalse(group_artifacts_ok(tmp, "Security", 6, 4))
        # corrupt ckpt
        (tmp / P.ckpt_npz_name("Security", 6)).write_bytes(b"junk")
        self.assertFalse(group_artifacts_ok(tmp, "Security", 6, 3))


class TestConstantsAndIsolation(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(P.EXPECTED_LN_TENSORS, 84)
        self.assertEqual(P.EXPECTED_LN_SCALARS, 55808)
        self.assertEqual(P.TRAIN_WINDOWS_TOTAL, 55836)
        self.assertEqual(P.VAL_WINDOWS_TOTAL, 7448)
        self.assertEqual(P.TEST_WINDOWS_TOTAL, 7388)
        self.assertEqual(P.TRAIN_STEPS_TOTAL, 236)
        self.assertEqual(P.FIXED_P,
                         {"Agriculture": 6, "Climate": 3, "Economy": 12,
                          "Energy": 4, "Environment": 7, "Health": 1,
                          "Security": 6, "SocialGood": 3, "Traffic": 12})
        self.assertEqual(P.PRECHECK, [("Energy", 48), ("Environment", 336),
                                      ("Health", 48), ("Security", 12)])
        self.assertEqual(P.EXP027_OVERALL_MSE, 1.6287341283427343)
        self.assertEqual(P.TATS_OVERALL_MSE, 1.6709424058596294)
        self.assertEqual(len(P.all_groups()), 36)
        for dom, h in P.all_groups():
            self.assertIn(dom, P.FIXED_P)

    def test_expected_steps(self):
        self.assertEqual(P.expected_steps(12), 1)
        self.assertEqual(P.expected_steps(256), 1)
        self.assertEqual(P.expected_steps(257), 2)
        self.assertEqual(P.expected_steps(0), 0)

    def test_package_import_isolation(self):
        base = Path(P.__file__).parent
        offenders = []
        for f in sorted(base.glob("*.py")):
            if f.name == "preflight.py":
                continue  # B2 cross-package comparison is intentional
            txt = f.read_text()
            if ("from repro_timemmd_common_v1" in txt
                    or "import repro_timemmd_common_v1" in txt):
                offenders.append(f.name)
        self.assertEqual(offenders, [])

    def test_val_npz_naming_distinct(self):
        self.assertEqual(P.val_npz_name("Health", 12), "val__Health__H12.npz")
        self.assertEqual(P.psearch_val_npz_name("Health", 1, 12),
                         "val__Health__P1__H12.npz")

    def test_numeric_npz_key_skips_strings(self):
        from repro_timemmd_common_ln_v1.replay_check import numeric_npz_key
        z = {"ln_names": np.array(["a.b", "c.d"]),
             "vision_model__norm__bias": np.ones(4, dtype=np.float32),
             "vision_model__norm__weight": np.ones(4, dtype=np.float32)}
        self.assertEqual(numeric_npz_key(z), "vision_model__norm__bias")
        with self.assertRaises(SystemExit):
            numeric_npz_key({"ln_names": np.array(["a.b"])})

    def test_weighted_batch_mean(self):
        from repro_timemmd_common_ln_v1.ln_train import weighted_batch_mean
        self.assertIsNone(weighted_batch_mean([], []))
        self.assertAlmostEqual(weighted_batch_mean([1.0, 2.0], [3, 1]),
                               1.25, places=12)
        exp = (0.4 * 256 + 0.8 * 7) / 263
        self.assertAlmostEqual(weighted_batch_mean([0.4, 0.8], [256, 7]),
                               exp, places=12)

    def test_replay_nc_paths_isolated_from_formal(self):
        from repro_timemmd_common_ln_v1.replay_one import output_paths
        out = Path("/tmp/whatever")
        gdir, npz, rep = output_paths(out, "Security", 6, None)
        self.assertTrue(str(gdir).endswith("runs/Security_H6"))
        self.assertNotIn("negative_control", str(gdir))
        self.assertEqual(npz.name, "replay__Security__H6.npz")
        self.assertEqual(rep.name, "replay_report__Security_H6.json")
        gdir2, npz2, rep2 = output_paths(out, "Security", 6, "variant_b")
        self.assertIn("negative_control", str(gdir2))
        self.assertNotEqual(gdir2, gdir)
        self.assertIn("variant_b", npz2.name)
        self.assertIn("variant_b", rep2.name)


if __name__ == "__main__":
    unittest.main()
