"""Training engine, sampler, metrics and data access on tiny synthetic data (CPU, a few seconds)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pathlens.config import AugCfg, ModelCfg, OptimCfg, RunCfg
from pathlens.evaluate import binary_metrics, bootstrap_ci, multiclass_metrics, paired_cluster_delta
from pathlens.identity import config_hash, file_sha256
from pathlens.model import load_for_inference
from pathlens.sampler import DeviceSampler
from pathlens.train import Interrupted, RunInputs, train_run


def _cfg(epochs=2, seed=0) -> RunCfg:
    return RunCfg(phase="p1", level="none", fold=-1, variant="clean", seed=seed,
                  model=ModelCfg(widths=(4, 8, 8, 8), num_classes=9),
                  optim=OptimCfg(epochs=epochs, batch_size=16, peak_lr=2e-3), aug=AugCfg(),
                  compile=False, amp_dtype="fp32")


def _inputs(n=96, n_test=32) -> RunInputs:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, n + n_test)
    base = np.array([[200, 80, 120], [120, 200, 80], [80, 120, 200]], dtype=np.uint8)[labels]
    imgs = np.clip(base[:, None, None, :] + rng.integers(-20, 20, (n + n_test, 64, 64, 3)), 0, 255).astype(np.uint8)
    return RunInputs(images=torch.from_numpy(imgs), labels=torch.from_numpy(labels).long(),
                     train_rows=torch.arange(n), test_rows=torch.arange(n, n + n_test),
                     row_ids=np.array([f"p1-x-{i:06d}" for i in range(n + n_test)]),
                     class_names=tuple(f"c{k}" for k in range(9)))


# ------------------------------------------------------------------ sampler

def test_sampler_covers_rows_once_and_is_seeded():
    rows = torch.arange(100, 150)
    a = list(DeviceSampler(rows, 8, torch.Generator().manual_seed(1), drop_last=False).epoch())
    b = list(DeviceSampler(rows, 8, torch.Generator().manual_seed(1), drop_last=False).epoch())
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert sorted(torch.cat(a).tolist()) == rows.tolist() and len(a) == 7
    s = DeviceSampler(rows, 8, torch.Generator().manual_seed(1), drop_last=True)
    assert len(s) == 6 and all(len(x) == 8 for x in s.epoch())
    with pytest.raises(ValueError):
        DeviceSampler(torch.arange(5).int(), 2, torch.Generator(), True)


# ------------------------------------------------------------------ training

def test_train_run_writes_everything_and_the_hashes_agree(tmp_path):
    cfg, inp = _cfg(), _inputs()
    res = train_run(cfg, inp, "cpu", tmp_path, lambda: False)
    for name in ("config.json", "train_log.jsonl", "model.pt", "model_card.json", "predictions.parquet",
                 "metrics.json", "DONE"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "ckpt.pt").exists()
    done = json.loads((tmp_path / "DONE").read_text())
    assert done["config_hash"] == config_hash(cfg) == res.config_hash
    assert done["model_hash"] == res.model_hash
    assert all(file_sha256(tmp_path / f) == h for f, h in done["sha256"].items())
    _, card = load_for_inference(tmp_path, "cpu")           # verifies model_hash against the card
    assert card.class_names[0] == "c0" and card.input_size == 64
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(df) == 32 and {"image_id", "label", "pred", "condition", "prob_0", "prob_8"} <= set(df.columns)
    assert df["prob_0"].dtype == np.float32 and np.allclose(df[[f"prob_{k}" for k in range(9)]].sum(axis=1), 1, atol=1e-4)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["steps"] == 2 * (96 // 16) and m["unstable"] is False and m["params"]["conv_weights"] > 0
    assert res.images_per_second > 0 and res.wall_seconds > 0


def test_train_run_learns_an_easy_problem(tmp_path):
    res = train_run(_cfg(epochs=6), _inputs(n=192), "cpu", tmp_path, lambda: False)
    acc = json.loads(res.metrics_path.read_text())["metrics"]["acc"]
    assert acc > 0.6                                            # 3 colour-coded classes, chance is 0.33


def test_finished_run_is_not_rerun_and_changed_config_is_refused(tmp_path):
    cfg, inp = _cfg(), _inputs()
    first = train_run(cfg, inp, "cpu", tmp_path, lambda: False)
    again = train_run(cfg, inp, "cpu", tmp_path, lambda: pytest.fail("must not train again"))
    assert again.model_hash == first.model_hash
    changed = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, peak_lr=1e-3))
    with pytest.raises(ValueError, match="config changed"):
        train_run(changed, inp, "cpu", tmp_path, lambda: False)


def test_interrupted_run_resumes_to_the_same_weights(tmp_path):
    cfg, inp = _cfg(epochs=3), _inputs()
    whole = train_run(cfg, inp, "cpu", tmp_path / "whole", lambda: False)
    with pytest.raises(Interrupted):
        train_run(cfg, inp, "cpu", tmp_path / "split", lambda: True)
    assert (tmp_path / "split" / "ckpt.pt").exists() and not (tmp_path / "split" / "DONE").exists()
    resumed = train_run(cfg, inp, "cpu", tmp_path / "split", lambda: False)
    assert resumed.model_hash == whole.model_hash
    assert not (tmp_path / "split" / "ckpt.pt").exists()


def test_twins_with_one_seed_see_the_same_sample_order(tmp_path):
    """Same seed -> same batches and augmentation draws; only the pixels differ."""
    inp = _inputs()
    a = train_run(_cfg(seed=3), inp, "cpu", tmp_path / "a", lambda: False)
    b = train_run(_cfg(seed=3), inp, "cpu", tmp_path / "b", lambda: False)
    c = train_run(_cfg(seed=4), inp, "cpu", tmp_path / "c", lambda: False)
    assert a.model_hash == b.model_hash != c.model_hash


# ------------------------------------------------------------------ metrics

def test_binary_and_multiclass_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.9])
    assert m["acc"] == 0.75 and m["auroc"] == 0.75 and m["n"] == 4 and m["n_pos"] == 2
    probs = np.eye(3)[[0, 1, 2, 2]] * 0.9 + 0.05
    mm = multiclass_metrics([0, 1, 2, 1], probs)
    assert mm["acc"] == 0.75 and mm["recall_per_class"] == [1.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        binary_metrics([1, 1], [0.2, 0.9])


def test_bootstrap_ci_brackets_the_estimate_and_counts_discards(rng):
    n = 400
    df = pd.DataFrame({"y": rng.integers(0, 2, n), "slide": np.repeat(np.arange(20), 20)})
    df["p"] = np.clip(df["y"] * 0.4 + rng.random(n) * 0.6, 0, 1)

    def auc(d):
        return binary_metrics(d["y"], d["p"])["auroc"]

    est, lo, hi = bootstrap_ci(df, auc, None, 200, rng)
    assert lo < est < hi
    _, clo, chi = bootstrap_ci(df, auc, "slide", 200, rng)
    assert clo < est < chi
    one = pd.DataFrame({"y": [1] * 99 + [0], "p": rng.random(100)})
    info: dict = {}
    with pytest.warns(UserWarning, match="discarded"):
        bootstrap_ci(one, auc, None, 200, rng, info=info)
    assert info["discarded"] > 2 and info["B"] == 200


def test_paired_cluster_delta_is_zero_for_identical_models_and_positive_for_better(rng):
    n = 400
    df = pd.DataFrame({"y": rng.integers(0, 2, n), "slide": np.repeat(np.arange(20), 20)})
    good = df.assign(p=np.clip(df["y"] * 0.5 + rng.random(n) * 0.5, 0, 1))
    bad = df.assign(p=rng.random(n))

    def auc(d):
        return binary_metrics(d["y"], d["p"])["auroc"]

    d0, lo0, hi0 = paired_cluster_delta(good, good, auc, "slide", 100, rng)
    assert d0 == 0 and lo0 == hi0 == 0
    d, lo, _ = paired_cluster_delta(bad, good, auc, "slide", 100, rng)
    assert d > 0.1 and lo > 0
    with pytest.raises(ValueError, match="same units"):
        paired_cluster_delta(good, good[good["slide"] > 0], auc, "slide", 10, rng)


# ------------------------------------------------------------------ data access

def test_load_p1_round_trip_and_refusals(tmp_path):
    from pathlens.data_p1 import load_p1
    n = {"train": 5, "val": 3, "test": 2}
    rows, rng = [], np.random.default_rng(0)
    for s, k in n.items():
        np.save(tmp_path / f"images_{s}.npy", rng.integers(0, 256, (k, 64, 64, 3), dtype=np.uint8))
        rows += [{"image_id": f"p1-{s}-{i:06d}", "split": s, "idx": i, "label": i % 3, "label_name": "x"}
                 for i in range(k)][::-1]                       # shuffled on purpose: load must sort by idx
    np.save(tmp_path / "images_train_poisoned.npy", np.zeros((5, 64, 64, 3), np.uint8))
    pd.DataFrame(rows).to_parquet(tmp_path / "meta.parquet")
    sp = load_p1("train", root=tmp_path)
    assert sp.images.shape == (5, 64, 64, 3) and sp.ids[0] == "p1-train-000000" and sp.labels.dtype == np.int64
    assert list(sp.labels) == [0, 1, 2, 0, 1]
    assert not load_p1("train", poisoned=True, root=tmp_path).images.any()
    with pytest.raises(ValueError, match="no poisoned test"):
        load_p1("test", poisoned=True, root=tmp_path)
    with pytest.raises(ValueError, match="split must be"):
        load_p1("dev", root=tmp_path)


def test_usable_cpus_respects_a_container_quota(tmp_path, monkeypatch):
    from pathlens import train
    monkeypatch.setattr(train.os, "process_cpu_count", lambda: 256)
    fake = tmp_path / "cpu.max"
    fake.write_text("3071999 100000\n")                                 # the rented box: about 30 CPUs
    real_path = train.Path
    monkeypatch.setattr(train, "Path", lambda p: fake if p == "/sys/fs/cgroup/cpu.max" else real_path(p))
    assert train.usable_cpus() == 30
    fake.write_text("max 100000\n")
    assert train.usable_cpus() == 256
    fake.write_text("garbage")
    assert train.usable_cpus() == 256
