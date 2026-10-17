"""Local CPU unit tests (no GPU, no real weights needed).

  python -m unittest repro_timemmd_common_v1.tests.test_local -v

Covers: data slicing/standardization/boundary gates, update_config geometry
vs hand formula (all table L/P/H combos), P selection tie->smaller, pred
comparator positive/negative controls, npz roundtrip (string identity),
adapter==direct forward on CPU, huggingface_hub offline import, guards
equality (tier-1 rejection).
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import bitwise_equal, guards_equal, \
    identity_equal, pred_close
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.preflight import geometry


def write_synth_protocol(tmp):
    """Tiny synthetic domain: N=40 rows, L=8, H=4; val 2 / test 2 windows."""
    rng = np.random.RandomState(7)
    ot = rng.randn(40) * 3 + 10
    dates = ["2001-%02d-01" % m for m in range(1, 41)]
    pdir = Path(tmp)
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "Synth.csv", "w") as f:
        f.write("date,OT\n")
        for d, v in zip(dates, ot):
            f.write("%s,%.17g\n" % (d, v))
    with open(pdir / "scaler.json", "w") as f:
        json.dump({"Synth": {"mean": float(ot.mean()),
                             "std": float(ot.std())}}, f)
    splits = {"Synth": {
        "field_map": {"date": "date", "target": "OT"},
        "N": 40, "L": 8, "H": [4], "freq": "M", "timef_dim": 2,
        "n_train": 30, "val_end": 35, "n_test": 5,
        "window_counts": {"4": {"train": 19, "val": 2, "test": 2}}}}
    (pdir / "windows").mkdir(exist_ok=True)
    with open(pdir / "windows" / "Synth_H4.csv", "w", newline="") as f:
        w = csv.writer(f)  # csv.writer handles embedded-quote escaping
        w.writerow(["domain", "H", "split", "s", "orig_row",
                    "input_date_range", "label_dates"])
        for split, lo, hi in (("train", 8, 27), ("val", 30, 32),
                              ("test", 35, 37)):
            for s in range(lo, hi):
                w.writerow(["Synth", 4, split, s, s,
                            json.dumps(dates[s - 8:s]),
                            json.dumps(dates[s:s + 4])])
    return pdir, splits, ot


class TestData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vts_data_")
        self.pdir, self.splits, self.ot = write_synth_protocol(self.tmp)
        self.dd = DomainData(self.pdir, self.splits, "Synth")

    def test_standardization_f64_to_f32(self):
        mean = float(self.ot.mean())
        std = float(self.ot.std())
        want = ((self.ot - mean) / std).astype(np.float32)
        np.testing.assert_array_equal(self.dd.series32, want)

    def test_slice_matches_manual(self):
        sp = self.dd.load_split(4, "test")
        self.assertEqual(sp["Y"].shape, (2, 4, 1))
        for i, s in enumerate(sp["s"]):
            np.testing.assert_array_equal(
                sp["X"][i], self.dd.series32[s - 8:s])
            np.testing.assert_array_equal(
                sp["Y"][i, :, 0], self.dd.series32[s:s + 4])

    def test_counts_gates(self):
        self.assertEqual(len(self.dd.load_split(4, "val")["X"]), 2)
        self.assertEqual(len(self.dd.load_split(4, "test")["X"]), 2)
        bad = json.loads(json.dumps(self.splits))
        bad["Synth"]["window_counts"]["4"]["val"] = 5
        with self.assertRaises(ValueError):
            DomainData(self.pdir, bad, "Synth").load_split(4, "val")

    def test_boundary_violation_raises(self):
        # forge a manifest row crossing the segment boundary
        wpath = Path(self.tmp) / "windows" / "Synth_H4.csv"
        lines = wpath.read_text().strip().split("\n")
        lines[-1] = lines[-1].replace(",36,36,", ",37,37,")
        # s=37 -> labels [37,41) crosses N=40; loader must fail via bounds
        wpath.write_text("\n".join(lines) + "\n")
        with self.assertRaises((ValueError, IndexError)):
            self.dd.load_split(4, "test")

    def test_identity_roundtrip_npz(self):
        import io
        sp = self.dd.load_split(4, "val")
        buf = io.BytesIO()
        np.savez_compressed(buf, s=sp["s"], orig_row=sp["orig_row"],
                            label_dates=sp["label_dates"])
        buf.seek(0)
        back = dict(np.load(buf))
        self.assertTrue(bitwise_equal(back["s"], sp["s"]))
        self.assertTrue(bitwise_equal(back["orig_row"], sp["orig_row"]))
        self.assertTrue(identity_equal(back["label_dates"],
                                       sp["label_dates"]))


class TestGeometry(unittest.TestCase):
    def test_all_table_combos_match_hand_formula(self):
        model = _tiny_model()
        for dom in P.ORDER:
            L = P.TABLE[dom]["L"]
            for h in P.TABLE[dom]["H"]:
                for p in P.TABLE[dom]["P"]:
                    model.update_config(context_len=L, pred_len=h,
                                        periodicity=p, norm_const=0.4,
                                        align_const=0.4,
                                        interpolation="bilinear")
                    g = geometry(L, h, p)
                    self.assertEqual(model.pad_left, g["pad_left"],
                                     (dom, h, p, "pad_left"))
                    self.assertEqual(model.pad_right, g["pad_right"],
                                     (dom, h, p, "pad_right"))
                    self.assertEqual(model.num_patch_input,
                                     g["num_patch_input"], (dom, h, p))
                    self.assertEqual(
                        model.input_resize.size,
                        (g["input_resize_h"], g["input_resize_w"]),
                        (dom, h, p))
                    self.assertEqual(
                        model.output_resize.size,
                        (g["output_resize_h"], g["output_resize_w"]),
                        (dom, h, p))

    def test_known_values(self):
        g = geometry(1056, 48, 52)
        self.assertEqual(g["pad_left"], 36)   # 1056 = 52*20+16
        self.assertEqual(g["pad_right"], 4)
        g = geometry(220, 12, 12)
        self.assertEqual(g["pad_left"], 8)
        self.assertEqual(g["pad_right"], 0)


class TestSelection(unittest.TestCase):
    def _fill(self, status, dom, fn):
        for p in P.TABLE[dom]["P"]:
            for h in P.TABLE[dom]["H"]:
                status["combos"][P.combo_id(dom, p, h)] = {"mse": fn(p)}

    def test_argmin_and_tie_smaller_p(self):
        from repro_timemmd_common_v1.p_search import select_periods
        status = {"combos": {}}
        for dom in P.ORDER:
            if dom == "Agriculture":
                self._fill(status, dom, lambda p: 1.0 if p != 6 else 0.5)
            else:
                self._fill(status, dom, lambda p: 2.0)
        sel = select_periods(status)
        self.assertEqual(sel["Agriculture"]["selected"], 6)
        # tie between all P -> smaller P
        status2 = {"combos": {}}
        for dom in P.ORDER:
            self._fill(status2, dom, lambda p: 0.7)
        sel2 = select_periods(status2)
        for dom in P.ORDER:
            self.assertEqual(sel2[dom]["selected"],
                             min(P.TABLE[dom]["P"]))


class TestComparators(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.a = rng.randn(5, 4).astype(np.float32)

    def test_identical_passes(self):
        c, mx, _ = pred_close(self.a, self.a, 1e-6, 1e-5)
        self.assertTrue(c)
        self.assertEqual(mx, 0.0)

    def test_subtol_passes(self):
        c, _, _ = pred_close(self.a + 1e-8, self.a, 1e-6, 1e-5)
        self.assertTrue(c)

    def test_perturbed_fails(self):
        c, mx, _ = pred_close(self.a + 0.01, self.a, 1e-6, 1e-5)
        self.assertFalse(c)
        self.assertGreater(mx, 0.009)  # f32 rounding may shave the delta

    def test_identity_string_array(self):
        a = np.array([["2021-01-0%d" % i] for i in (1, 2, 3)], dtype="U10")
        self.assertTrue(identity_equal(a, a.copy()))
        self.assertFalse(identity_equal(a, a[::-1].copy()))

    def test_bitwise_uint_view(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        self.assertTrue(bitwise_equal(a, a.copy()))
        self.assertFalse(bitwise_equal(a, a + 1e-5))


class TestGuards(unittest.TestCase):
    def test_equal_and_reject(self):
        a = {"k1": "v", "code_md5": {"x.py": "1"}, "batch": 256}
        b = dict(a)
        ok, diff = guards_equal(a, b)
        self.assertTrue(ok)
        b["batch"] = 128
        ok, diff = guards_equal(a, b)
        self.assertFalse(ok)
        self.assertEqual(diff, ["batch"])

    def test_code_md5_reject(self):
        a = {"code_md5": {"x.py": "1"}}
        b = {"code_md5": {"x.py": "2"}}
        ok, diff = guards_equal(a, b)
        self.assertFalse(ok)
        self.assertEqual(diff, ["code_md5"])


class TestPsearchCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vts_psearch_")
        self.pdir, self.splits, _ = write_synth_protocol(self.tmp)
        self.dd = DomainData(self.pdir, self.splits, "Synth")
        self.pinfo = {"weights_sha256": "w" * 64}
        from repro_timemmd_common_v1.p_search import combo_fingerprint
        self.combo_fingerprint = combo_fingerprint

    def _write_npz(self, path, sp, pred):
        np.savez_compressed(path, pred=pred, truth=sp["Y"], s=sp["s"],
                            orig_row=sp["orig_row"],
                            label_dates=sp["label_dates"],
                            window_index=sp["window_index"])

    def test_fingerprint_stable_and_discriminating(self):
        a = self.combo_fingerprint(self.pinfo, "Synth", 3, 4)
        b = self.combo_fingerprint(self.pinfo, "Synth", 3, 4)
        self.assertEqual(a, b)
        self.assertNotEqual(a["weights_sha256"], "")
        c = self.combo_fingerprint(self.pinfo, "Synth", 3, 8)
        self.assertNotEqual(a, c)
        d = self.combo_fingerprint({"weights_sha256": "z" * 64},
                                   "Synth", 3, 4)
        self.assertNotEqual(a, d)

    def test_verify_combo_arrays_positive(self):
        from repro_timemmd_common_v1.p_search import verify_combo_arrays
        sp = self.dd.load_split(4, "val")
        pred = np.zeros((2, 4, 1), dtype=np.float32)
        npz = Path(self.tmp) / "val__Synth__P3__H4.npz"
        self._write_npz(npz, sp, pred)
        chk = verify_combo_arrays(npz, sp)
        self.assertTrue(chk["ok"], chk["reason"])
        self.assertEqual(chk["windows"], 2)

    def test_verify_combo_array_rejects_tampering(self):
        from repro_timemmd_common_v1.p_search import verify_combo_arrays
        sp = self.dd.load_split(4, "val")
        pred = np.zeros((2, 4, 1), dtype=np.float32)
        npz = Path(self.tmp) / "val__Synth__P3__H4.npz"
        self._write_npz(npz, sp, pred)
        # tampered truth
        arr = dict(np.load(npz))
        arr["truth"] = arr["truth"] + np.float32(0.5)
        np.savez_compressed(npz, **arr)
        chk = verify_combo_arrays(npz, sp)
        self.assertFalse(chk["ok"])
        self.assertIn("truth", chk["reason"])
        # non-permutation window_index
        arr = dict(np.load(npz))
        arr["truth"] = sp["Y"]
        arr["window_index"] = np.array([0, 0])
        np.savez_compressed(npz, **arr)
        chk = verify_combo_arrays(npz, sp)
        self.assertFalse(chk["ok"])
        self.assertIn("permutation", chk["reason"])
        # missing key
        arr.pop("s")
        np.savez_compressed(npz, **arr)
        chk = verify_combo_arrays(npz, sp)
        self.assertFalse(chk["ok"])
        self.assertIn("missing key", chk["reason"])


class TestAdapterDirect(unittest.TestCase):
    def test_adapter_equals_direct_cpu(self):
        import torch
        from repro_timemmd_common_v1.adapter import build_visionts, \
            configure, forward_batch
        from repro_timemmd_common_v1.common import seed_everything
        model = build_visionts(load_ckpt=False)
        model.eval()
        configure(model, 96, 12, 4)
        seed_everything(2021)
        x = torch.randn(5, 96, 1)
        ya = forward_batch(model, x)
        with torch.inference_mode():
            yd = model(x, export_image=False, fp64=False)
        np.testing.assert_array_equal(ya.numpy(), yd.numpy())
        self.assertEqual(ya.shape, (5, 12, 1))

    def test_frozen_when_finetune_none(self):
        from repro_timemmd_common_v1.adapter import build_visionts
        model = build_visionts(load_ckpt=False)
        for prm in model.vision_model.parameters():
            self.assertFalse(prm.requires_grad)


class TestOfflineImport(unittest.TestCase):
    def test_huggingface_hub_importable_offline(self):
        code = ("import os; os.environ['HF_HUB_OFFLINE']='1'; "
                "import huggingface_hub; print(huggingface_hub.__version__)")
        import subprocess
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


def _tiny_model():
    from repro_timemmd_common_v1.adapter import build_visionts
    return build_visionts(load_ckpt=False)


if __name__ == "__main__":
    unittest.main()
