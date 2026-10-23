"""PairedStemDataset/PairedBatchSampler: real+fake same stem, deterministic crops, batch mixes."""
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _common
import train_lora as TL


def _write(root, names):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        p = root / n
        g = np.random.default_rng(int(hashlib.md5(n.encode()).hexdigest()[:8], 16))
        Image.fromarray(g.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
        out[p.stem] = str(p)
    return out


def _setup(td, n_stems):
    real = _write(Path(td) / "real", [f"stem{i}.png" for i in range(n_stems)])
    fake = _write(Path(td) / "fake", [f"stem{i}.png" for i in range(n_stems)])
    ids = sorted(real)
    return ids, real, [fake], [str(Path(td) / "fake")]


def _cfg():
    TL.CFG.update({"epochs": 4, "pair_real_prob": 0.5,
                   "degrade_max_rounds": 0, "degrade_round_probs": [1.0]})


def test_pairs_share_stem_and_labels_alternate():
    with tempfile.TemporaryDirectory() as td:
        ids, real, fake_maps, _ = _setup(td, 6)
        _cfg()
        ds = TL.PairedStemDataset(ids, 64, real, fake_maps)
        ds.set_epoch(1)
        for i in range(len(ids)):
            assert (ds[2 * i][2], ds[2 * i + 1][2]) == (0, 1)


def test_crop_is_deterministic_per_stem_and_epoch():
    with tempfile.TemporaryDirectory() as td:
        ids, real, fake_maps, _ = _setup(td, 2)
        _cfg()
        a = TL.PairedStemDataset(ids, 64, real, fake_maps)
        b = TL.PairedStemDataset(ids, 64, real, fake_maps)
        a.set_epoch(2)
        b.set_epoch(2)
        for i in range(4):  # 2 stems x (real, fake)
            assert torch.equal(a[i][0], b[i][0]), i
        a.set_epoch(3)
        assert not torch.equal(a[0][0], b[0][0])  # different epoch -> different crop path


def test_sampler_batches_mix_other_fraction():
    with tempfile.TemporaryDirectory() as td:
        ids, real, fake_maps, _ = _setup(td, 12)
        _cfg()
        ds = TL.PairedStemDataset(ids, 64, real, fake_maps)
        s = TL.PairedBatchSampler(len(ds), n_other=40, batch_size=8, other_frac=0.25, seed=0)
        assert len(s) == 4
        stems, others = set(), 0
        for b in s:
            assert len(b) == 8 and len(set(b)) == 8
            others += sum(1 for i in b if i >= len(ds))
            for i in b:
                if i < len(ds):
                    stems.add(i // 2)
        assert others == 2 * len(s) and len(stems) == len(ids)
        s.set_epoch(1)
        assert next(iter(s)) == next(iter(s))  # deterministic within an epoch


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK paired dataset/sampler")
