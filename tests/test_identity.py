import json

import torch

from pathlens.config import P1_MODEL, P2_MODEL, p1_run_cfg
from pathlens.identity import canonical_json, config_hash, file_sha256, model_hash, run_id
from pathlens.model import build_model


def test_canonical_json_is_sorted_compact_bytes():
    out = canonical_json({"b": 1, "a": [1.5, (2, 3)], "c": {"z": 0.1, "y": 2e-3}})
    assert isinstance(out, bytes)
    assert out == b'{"a":[1.5,[2,3]],"b":1,"c":{"y":0.002,"z":0.1}}'


def test_config_hash_is_sha256_prefix_of_canonical_asdict():
    import dataclasses
    import hashlib
    cfg = p1_run_cfg("clean")
    assert config_hash(cfg) == hashlib.sha256(canonical_json(dataclasses.asdict(cfg))).hexdigest()[:12]


def test_run_id_matches_doc_example():
    from pathlens.config import expand_queue
    ids = {run_id(r) for r in expand_queue("configs/phase2_gpu.yaml")}
    assert "p2-hospital-f2-aug-s0" in ids
    assert run_id(p1_run_cfg("poisoned")) == "p1-none-f-1-poisoned-s0"


def test_model_hash_on_real_models_with_batchnorm_counters():
    for cfg in (P1_MODEL, P2_MODEL):
        sd = build_model(cfg).state_dict()
        assert any(v.ndim == 0 for v in sd.values())         # num_batches_tracked
        h = model_hash(sd)
        assert len(h) == 12 and h == model_hash(dict(reversed(list(sd.items()))))


def test_model_hash_sensitivity_and_dtype_cases():
    sd = build_model(P1_MODEL).state_dict()
    h = model_hash(sd)
    sd2 = {k: v.clone() for k, v in sd.items()}
    k = next(k for k, v in sd2.items() if v.is_floating_point())
    sd2[k].view(-1)[0] += 1e-3
    assert model_hash(sd2) != h
    sd3 = {k: v.clone() for k, v in sd.items()}
    sd3["bn_counter"] = torch.tensor(5)
    assert model_hash(sd3) != h
    # bfloat16, empty tensors and non-contiguous tensors do not crash and are deterministic
    odd = {"a": torch.randn(3, 4).bfloat16(), "b": torch.empty(0), "c": torch.randn(4, 3).t(),
           "d": torch.tensor(7)}
    assert model_hash(odd) == model_hash({k: v.clone() for k, v in odd.items()})
    assert model_hash({"x": torch.zeros(2, 3)}) != model_hash({"x": torch.zeros(3, 2)})
    assert model_hash({"x": torch.zeros(2)}) != model_hash({"y": torch.zeros(2)})


def test_model_hash_independent_of_save_format(tmp_path):
    sd = build_model(P1_MODEL).state_dict()
    torch.save(sd, tmp_path / "a.pt")
    torch.save(dict(reversed(list(sd.items()))), tmp_path / "b.pt", _use_new_zipfile_serialization=False)
    a = torch.load(tmp_path / "a.pt", weights_only=True)
    b = torch.load(tmp_path / "b.pt", weights_only=True)
    assert model_hash(a) == model_hash(b) == model_hash(sd)


def test_file_sha256_streams_across_blocks(tmp_path):
    import hashlib
    data = bytes(range(256)) * (40_000)                     # ~10 MB: more than one 8 MB block
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    assert file_sha256(p) == hashlib.sha256(data).hexdigest()
    assert json.dumps(file_sha256(p))                       # plain str
