import pickle
from copy import deepcopy

import numpy as np
import pytest
import torch

from nullscape.inference.sampler import TerrainSampler
from nullscape.metrics.quality import CONDITION_KEYS
from nullscape.models.diffusion import (DiffusionConfig, GaussianDiffusion, cosine_alphas_cumprod,
                                        timestep_schedule)
from nullscape.models.ema import EMA
from nullscape.models.unet import ConditionEmbedding, UNet, UNetConfig, count_parameters
from nullscape.terrain.archetypes import ARCHETYPES
from nullscape.train.trainer import TensorBatcher, save_checkpoint
from nullscape.utils.seed import seed_everything
from nullscape.world import WorldSpec

TINY = UNetConfig(image_size=16, base_channels=32, channel_mults=(1, 2), num_res_blocks=1,
                  attention_resolutions=(8,), num_conditions=5, num_classes=6, dropout=0.0)
W16 = WorldSpec(resolution=16)
DATASET_META = {
    "world": W16.to_dict(),
    "condition_keys": list(CONDITION_KEYS),
    "condition_stats": {"mean": [0.0] * len(CONDITION_KEYS), "std": [1.0] * len(CONDITION_KEYS)},
    "archetypes": list(ARCHETYPES),
}
K, C = len(CONDITION_KEYS), len(ARCHETYPES)


def _tiny_inputs(b: int = 2):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, 1, TINY.image_size, TINY.image_size, generator=g)
    t = torch.randint(0, 1000, (b,), generator=g)
    cond = torch.randn(b, K, generator=g)
    known = torch.ones(b, K, dtype=torch.bool)
    label = torch.randint(0, C, (b,), generator=g)
    return x, t, cond, known, label


def test_unet_output_shape():
    for res in (16, 32):
        cfg = UNetConfig(**{**TINY.to_dict(), "image_size": res})
        model = UNet(cfg)
        x, t, cond, known, label = _tiny_inputs()
        x = torch.randn(2, 1, res, res)
        assert model(x, t, cond, known, label).shape == x.shape


def test_default_unet_param_count():
    n = count_parameters(UNet(UNetConfig()))
    print(f"\ndefault UNetConfig parameters: {n} ({n / 1e6:.2f}M)")
    assert 10e6 < n < 25e6


def test_fresh_unet_outputs_zeros():
    model = UNet(TINY).eval()
    x, t, cond, known, label = _tiny_inputs()
    out = model(x, t, cond, known, label)
    assert torch.equal(out, torch.zeros_like(out))


def test_condition_embedding_unknown_is_ignored():
    emb = ConditionEmbedding(K, C, 64).eval()
    g = torch.Generator().manual_seed(1)
    cond = torch.randn(4, K, generator=g)
    known = torch.tensor([[True, False, True, True, False]] * 4)
    label = torch.tensor([0, 1, 2, C])  # C = unknown label, must be valid
    base = emb(cond, known, label)

    changed = cond.clone()
    changed[:, 1] = 99.0
    changed[:, 4] = -50.0
    assert torch.equal(emb(changed, known, label), base)

    changed2 = cond.clone()
    changed2[:, 0] += 1.0
    assert not torch.equal(emb(changed2, known, label), base)


def test_cosine_schedule_decreasing():
    ab = cosine_alphas_cumprod(1000)
    assert (ab[1:] < ab[:-1]).all()
    assert (ab > 0).all() and (ab <= 1.0).all()


def test_timestep_schedule_uniform_backwards_compatible():
    ts = timestep_schedule(1000, 50, "uniform")
    assert torch.equal(ts, torch.linspace(999, 0, 50).round().long())


def test_timestep_schedule_quadratic():
    ts = timestep_schedule(1000, 50, "quadratic")
    assert (ts[1:] < ts[:-1]).all()          # strictly decreasing
    assert ts[0].item() == 999 and ts[-1].item() == 0
    assert len(ts) <= 50                      # de-duplicated
    assert (ts[-5:] <= 10).all()              # fine detail resolved at low noise


def test_timestep_schedule_invalid():
    with pytest.raises(ValueError):
        timestep_schedule(1000, 50, "cubic")


def test_v_prediction_reconstruction():
    diff = GaussianDiffusion(DiffusionConfig(timesteps=200), num_classes=C)
    g = torch.Generator().manual_seed(2)
    x0 = torch.randn(4, 1, 16, 16, generator=g)
    eps = torch.randn_like(x0)
    t = torch.randint(0, 200, (4,), generator=g)
    x_t = diff.q_sample(x0, t, eps)
    ab = diff.alphas_cumprod[t].view(-1, 1, 1, 1)
    v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
    x0_hat = ab.sqrt() * x_t - (1 - ab).sqrt() * v
    assert torch.allclose(x0_hat, x0, atol=1e-5)


def test_drop_conditions_probabilities():
    known = torch.ones(8, K, dtype=torch.bool)
    label = torch.arange(8) % C
    d0 = GaussianDiffusion(DiffusionConfig(timesteps=100, p_drop_property=0.0, p_drop_label=0.0,
                                           p_drop_all=0.0), C)
    k2, l2 = d0.drop_conditions(known.clone(), label.clone())
    assert torch.equal(k2, known) and torch.equal(l2, label)

    d1 = GaussianDiffusion(DiffusionConfig(timesteps=100, p_drop_all=1.0), C)
    k3, l3 = d1.drop_conditions(known.clone(), label.clone())
    assert not k3.any() and (l3 == C).all()


def test_overfit_tiny_model():
    seed_everything(0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 1, 16, 16, generator=g).clamp(-1, 1)
    cond = torch.randn(8, K, generator=g)
    label = torch.randint(0, C, (8,), generator=g)
    model = UNet(TINY)
    diff = GaussianDiffusion(DiffusionConfig(timesteps=200), C)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(200):
        loss = diff.loss(model, x, cond, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    print(f"\noverfit: first20={first:.4f} last20={last:.4f}")
    assert last < 0.5 * first


def test_loss_deterministic_with_fixed_noise():
    model = UNet(TINY).eval()
    diff = GaussianDiffusion(DiffusionConfig(timesteps=200), C)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 1, 16, 16, generator=g)
    cond = torch.randn(4, K, generator=g)
    label = torch.randint(0, C, (4,), generator=g)
    t = torch.randint(0, 200, (4,), generator=g)
    noise = torch.randn_like(x)
    a = diff.loss(model, x, cond, label, t=t, noise=noise, drop=False)
    b = diff.loss(model, x, cond, label, t=t, noise=noise, drop=False)
    assert torch.equal(a, b)


def test_ema_tracks_and_state_roundtrip():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.9, warmup_steps=10)
    with torch.no_grad():
        model.weight.add_(1.0)
    p0 = ema.shadow.weight.detach().clone()
    ema.update(model)
    # EMA uses lerp weight 1-d on the live params: shadow = shadow*d + model*(1-d)
    d = min(ema.decay, (1 + ema.num_updates) / (ema.warmup_steps + ema.num_updates))
    expected = p0 * d + model.weight.detach() * (1 - d)
    assert torch.allclose(ema.shadow.weight, expected)

    ema2 = EMA(torch.nn.Linear(4, 4))
    ema2.load_state_dict(ema.state_dict())
    assert torch.equal(ema2.shadow.weight, ema.shadow.weight)
    assert ema2.num_updates == ema.num_updates


@pytest.fixture(scope="module")
def sampler() -> TerrainSampler:
    g = torch.Generator().manual_seed(7)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = UNet(TINY)
    # residual branches and the out conv are zero-init, which severs every path
    # (including conditioning) to the output; perturb all params slightly so the
    # randomly initialized model actually responds to inputs and conditions
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    diff = GaussianDiffusion(DiffusionConfig(timesteps=32), C)
    return TerrainSampler(model, diff, DATASET_META, device="cpu")


def test_sampler_batch_size_invariance(sampler):
    a = sampler.sample(n=4, seed=3, steps=4, batch_size=4)
    b = sampler.sample(n=4, seed=3, steps=4, batch_size=1)
    assert np.allclose(a, b, atol=1e-5)
    c = sampler.sample(n=4, seed=3, steps=4, batch_size=4, eta=0.5)
    d = sampler.sample(n=4, seed=3, steps=4, batch_size=1, eta=0.5)
    assert np.allclose(c, d, atol=1e-5)


def test_sampler_quadratic_spacing(sampler):
    a = sampler.sample(n=4, seed=3, steps=4, batch_size=4, spacing="quadratic")
    b = sampler.sample(n=4, seed=3, steps=4, batch_size=1, spacing="quadratic")
    assert np.allclose(a, b, atol=1e-5)
    u = sampler.sample(n=4, seed=3, steps=4, batch_size=4, spacing="uniform")
    assert not np.allclose(a, u)


def test_sampler_output_contract_and_seed(sampler):
    a = sampler.sample(n=4, seed=3, steps=4)
    assert a.dtype == np.float32 and a.shape == (4, 16, 16)
    assert a.min() >= 0.0 and a.max() <= 1.0
    b = sampler.sample(n=4, seed=4, steps=4)
    assert not np.allclose(a, b)


def test_sampler_conditioning_errors_and_guidance(sampler):
    with pytest.raises(KeyError):
        sampler.sample(n=1, seed=0, steps=2, properties={"not_a_key": 1.0})
    with pytest.raises(ValueError):
        sampler.sample(n=1, seed=0, steps=2, archetype="volcano")
    out = sampler.sample(n=2, seed=0, steps=4, labels=[-1, -1])
    assert out.shape == (2, 16, 16)
    # guidance only changes output when conditioning is actually given
    kwargs = dict(n=2, seed=5, steps=4, archetype="mountains", properties={"relief": 0.4})
    g1 = sampler.sample(guidance=1.0, **kwargs)
    g3 = sampler.sample(guidance=3.0, **kwargs)
    assert not np.allclose(g1, g3)


def test_checkpoint_roundtrip(sampler, tmp_path):
    model = deepcopy(sampler.model)
    diffusion = sampler.diffusion
    ema = EMA(model, decay=0.9, warmup_steps=0)
    w_ema = model.out[-1].weight.detach().clone()  # shadow == current weights
    with torch.no_grad():
        model.out[-1].weight.add_(0.5)  # raw weights now differ from EMA
    w_raw = model.out[-1].weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters())
    cfg = {"_diffusion_config": diffusion.cfg.to_dict(), "train": {}}
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, ema, opt, 7, cfg, DATASET_META)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("dataset", "unet_config", "diffusion_config", "step"):
        assert key in ck
    assert ck["step"] == 7

    s_ema = TerrainSampler.from_checkpoint(path, use_ema=True, device="cpu")
    s_raw = TerrainSampler.from_checkpoint(path, use_ema=False, device="cpu")
    assert torch.equal(s_ema.model.out[-1].weight, w_ema)
    assert torch.equal(s_raw.model.out[-1].weight, w_raw)
    assert s_ema.checkpoint_info["step"] == 7


def test_release_checkpoint_loads_without_pickle_and_samples_identically(sampler, tmp_path):
    from nullscape.utils.checkpoint import export_release, file_sha256, load_checkpoint

    model = deepcopy(sampler.model)
    cfg = {"_diffusion_config": sampler.diffusion.cfg.to_dict(), "name": "tiny",
           "train": {"init_from": str(tmp_path / "private" / "parent.pt"), "artifacts_dir": None}}
    source = tmp_path / "ck.pt"
    state = {"version": 1, "rng": {"numpy": np.random.get_state()}}
    save_checkpoint(source, model, EMA(model), torch.optim.AdamW(model.parameters()), 5, cfg,
                    {**DATASET_META, "root": str(tmp_path / "private" / "data")}, {"training_state": state})
    with pytest.raises(pickle.UnpicklingError):
        torch.load(source, map_location="cpu", weights_only=True)  # numpy RNG state needs the allowlist
    assert load_checkpoint(source)["training_state"]["rng"]["numpy"][1].dtype == np.uint32

    out = tmp_path / "release" / "tiny-1.pt"
    info = export_release(source, out, name="Tiny", version="1.0.0", license="Apache-2.0")
    assert info["sha256"] == file_sha256(out) and info["source"]["sha256"] == file_sha256(source)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    assert not {"model", "optimizer", "training_state"} & set(ck)
    assert "root" not in ck["dataset"] and ck["train_config"]["train"]["init_from"] == "parent.pt"
    assert "private" not in repr({k: v for k, v in ck.items() if k != "ema"})

    a = TerrainSampler.from_checkpoint(source, device="cpu").sample(n=2, seed=3, steps=3, archetype="hills")
    released = TerrainSampler.from_checkpoint(out, device="cpu")
    assert np.array_equal(a, released.sample(n=2, seed=3, steps=3, archetype="hills"))
    assert released.checkpoint_info["release"] == "Tiny 1.0.0"
    with pytest.raises(ValueError, match="EMA weights only"):
        TerrainSampler.from_checkpoint(out, device="cpu", use_ema=False)
    with pytest.raises(FileExistsError):
        export_release(source, out, name="Tiny", version="1.0.0", license="Apache-2.0")


def test_tensor_batcher():
    rng = np.random.default_rng(0)
    heights = (rng.random((4, 16, 16)) * 65535).astype(np.uint16)
    cond = rng.standard_normal((4, K)).astype(np.float32)
    labels = np.array([0, 1, 2, 3])
    b = TensorBatcher(heights, cond, labels, torch.device("cpu"), on_device=False, seed=0)
    x, c, y = b.sample(8, augment=True)
    assert x.shape == (8, 1, 16, 16)
    assert float(x.min()) >= -1.0 - 1e-6 and float(x.max()) <= 1.0 + 1e-6
    assert c.shape == (8, K) and y.shape == (8,)
    # dihedral augmentation must preserve each sample's value multiset
    expected = [np.sort((m.astype(np.float32) / 65535.0 * 2.0 - 1.0).ravel()) for m in heights]
    for j in range(8):
        s = np.sort(x[j, 0].numpy().ravel())
        assert any(np.array_equal(s, e) for e in expected), f"sample {j} not a dihedral copy"


def test_heightparam_roundtrip():
    from nullscape.models import heightparam

    rng = np.random.default_rng(1)
    h = rng.random((3, 16, 16)).astype(np.float32) * 0.1 + 0.4
    m, r = h.mean((1, 2))[:, None, None], (np.percentile(h, 98, (1, 2)) - np.percentile(h, 2, (1, 2)))[:, None, None]
    x = heightparam.encode(h, m, r, "relative")
    assert np.allclose(heightparam.decode(x, m, r, "relative"), h, atol=1e-6)
    assert x.std() > 5 * (h * 2 - 1).std()  # a flat map fills far more of the model's range than absolute
    t = torch.from_numpy(h)
    assert torch.allclose(heightparam.decode(heightparam.encode(t, torch.from_numpy(m), torch.from_numpy(r),
                                                                "relative"), torch.from_numpy(m),
                                             torch.from_numpy(r), "relative"), t, atol=1e-6)


def test_tensor_batcher_relative():
    rng = np.random.default_rng(0)
    heights = (rng.random((4, 16, 16)) * 65535).astype(np.uint16)
    h = heights.astype(np.float32) / 65535
    place = np.stack([h.mean((1, 2)), np.percentile(h, 98, (1, 2)) - np.percentile(h, 2, (1, 2))], 1)
    b = TensorBatcher(heights, np.zeros((4, K), np.float32), np.arange(4), torch.device("cpu"), on_device=False,
                      seed=0, placement=place)
    x, _, _ = b.sample(16, augment=True)
    assert abs(float(x.mean())) < 0.05  # each map is centred on its own mean


def test_relative_sampler_places_and_fills(sampler):
    from nullscape.models import heightparam

    bank = {"conditions": [[0.3, 0.2, 10.0, 0.1, 4.0], [0.5, 0.05, 3.0, 0.0, 5.0]] * 6,
            "labels": [c for c in range(C) for _ in range(2)]}
    meta = {**DATASET_META, "height_param": heightparam.spec("relative"), "prior_bank": bank}
    rs = TerrainSampler(sampler.model, sampler.diffusion, meta, device="cpu")
    # an untrained model's shape can be anything within the x0 clamp (+-8); with relief 0.02 the decoded map
    # must still sit within 8 * 0.02 / 2 = 0.08 of the requested mean elevation
    a = rs.sample(n=2, seed=1, steps=4, archetype="plains", properties={"mean_elevation": 0.4, "relief": 0.02})
    assert a.shape == (2, 16, 16) and a.min() >= 0 and a.max() <= 1
    assert abs(float(a.mean()) - 0.4) <= 0.08 + 1e-6
    b = rs.sample(n=3, seed=2, steps=4, archetype="hills")  # placement borrowed from the prior bank
    assert b.shape == (3, 16, 16) and np.isfinite(b).all()
    with pytest.raises(ValueError):
        TerrainSampler(sampler.model, sampler.diffusion, {**meta, "prior_bank": None}, device="cpu").sample(
            n=1, seed=0, steps=2)


@pytest.mark.parametrize("label", [C, 2])
@pytest.mark.parametrize("water_known", [False, True])
def test_relative_prior_ties_cover_eligible_bank(sampler, label, water_known):
    from nullscape.models import heightparam

    bank = np.zeros((64 * C, K), dtype=np.float32)
    bank[:, 0] = np.linspace(0.1, 0.7, len(bank))
    bank[:, 1] = 0.1
    labels = np.repeat(np.arange(C), 64)
    meta = {**DATASET_META, "height_param": heightparam.spec("relative"),
            "prior_bank": {"conditions": bank, "labels": labels}}
    rs = TerrainSampler(sampler.model, sampler.diffusion, meta, device="cpu")
    raw = np.zeros((512, K), dtype=np.float32)
    known = np.zeros_like(raw, dtype=bool)
    known[:, 3] = water_known
    lab = np.full(len(raw), label)
    seeds = list(range(len(raw)))
    filled, filled_known = rs._fill_placement(raw, known, lab, seeds)
    assert len(np.unique(filled[:, 0])) > 32
    chosen = np.searchsorted(bank[:, 0], filled[:, 0])
    assert set(labels[chosen]) == (set(range(C)) if label == C else {label})
    parts = [rs._fill_placement(raw[i:i + 64], known[i:i + 64], lab[i:i + 64], seeds[i:i + 64])[0]
             for i in range(0, len(raw), 64)]
    assert np.array_equal(filled, np.concatenate(parts))
    assert filled_known[:, :2].all() and not known[:, :2].any()
    assert np.array_equal(raw, np.zeros_like(raw))


def test_sampling_defaults_and_overrides(sampler):
    official = sampler.resolve_sampling()
    assert official["steps"] == 50 and official["spacing"] == "quadratic"
    assert official["guidance"] == 2.0 and official["eta"] == 0.0
    meta = {**DATASET_META, "sampling": {"steps": 7, "guidance": 1.25, "spacing": "uniform"}}
    s = TerrainSampler(sampler.model, sampler.diffusion, meta, device="cpu")
    assert s.resolve_sampling()["steps"] == 7
    assert s.resolve_sampling(steps=4, eta=0.3)["eta"] == 0.3
    assert s.resolve_sampling(steps=4)["guidance"] == 1.25
    a = s.sample(n=1, seed=19)
    b = s.sample(n=1, seed=19, steps=7, guidance=1.25, spacing="uniform", eta=0.0)
    assert np.array_equal(a, b)


def test_checkpoint_sampling_preset_roundtrip(sampler, tmp_path):
    cfg = {"_diffusion_config": sampler.diffusion.cfg.to_dict(), "train": {},
           "sampling": {"steps": 5, "guidance": 1.3, "spacing": "uniform"}}
    path = tmp_path / "preset.pt"
    save_checkpoint(path, sampler.model, EMA(sampler.model), torch.optim.AdamW(sampler.model.parameters()),
                    1, cfg, DATASET_META)
    loaded = TerrainSampler.from_checkpoint(path, device="cpu")
    assert loaded.resolve_sampling()["steps"] == 5
    assert loaded.resolve_sampling()["guidance"] == 1.3


def test_guidance_interval_uses_normalized_noise_time(sampler, monkeypatch):
    calls = []

    def guided(model, x, t, cond, known, label, guidance):
        calls.append((int(t[0]), guidance))
        return torch.zeros_like(x)

    monkeypatch.setattr(sampler.diffusion, "_guided_v", guided)
    sampler.sample(n=1, steps=5, spacing="uniform", guidance=3.0, guidance_interval=(0.2, 0.8))
    assert [g for _, g in calls] == [1.0, 3.0, 3.0, 3.0, 1.0]
    assert calls[0][0] == 31 and calls[-1][0] == 0


def test_guidance_full_interval_is_compatible(sampler):
    kw = dict(n=2, steps=5, seed=15, archetype="ridges", guidance=2.0)
    assert np.array_equal(sampler.sample(**kw), sampler.sample(**kw, guidance_interval=(0.0, 1.0)))
    for interval in [(-0.1, 0.8), (0.9, 0.2), (0.0, float("nan")), (0.0, 1.1)]:
        with pytest.raises(ValueError):
            sampler.sample(**kw, guidance_interval=interval)


def test_weighted_diffusion_loss(sampler):
    x, t, cond, known, label = _tiny_inputs()
    t = t % sampler.diffusion.cfg.timesteps
    noise = torch.randn_like(x)
    d = sampler.diffusion
    pieces = [d.loss(sampler.model, x[i:i + 1], cond[i:i + 1], label[i:i + 1],
                     t=t[i:i + 1], noise=noise[i:i + 1], drop=False) for i in range(2)]
    got = d.loss(sampler.model, x, cond, label, t=t, noise=noise, drop=False,
                 sample_weight=torch.tensor([1.0, 3.0]))
    assert torch.allclose(got, (pieces[0] + 3 * pieces[1]) / 4, atol=1e-5)
    plain = d.loss(sampler.model, x, cond, label, t=t, noise=noise, drop=False)
    equal = d.loss(sampler.model, x, cond, label, t=t, noise=noise, drop=False,
                   sample_weight=torch.ones(2))
    assert torch.equal(plain, equal)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_inpainting_preserves_heights_and_batch_invariance(sampler, relative, eta):
    from nullscape.models import heightparam

    meta = {**DATASET_META, "height_param": heightparam.spec("relative" if relative else "absolute")}
    s = TerrainSampler(sampler.model, sampler.diffusion, meta, device="cpu")
    reference = np.linspace(0.1, 0.8, 256, dtype=np.float32).reshape(16, 16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, :4] = True
    kw = dict(n=2, seed=3, steps=4, eta=eta, reference=reference, preserve_mask=mask,
              properties={"mean_elevation": 0.4, "relief": 0.3})
    a = s.sample(**kw, batch_size=2)
    b = s.sample(**kw, batch_size=1)
    assert np.array_equal(a[:, mask], np.broadcast_to(reference[mask], a[:, mask].shape))
    assert np.allclose(a, b, atol=1e-5)
    assert not np.array_equal(a[0, ~mask], a[1, ~mask])
    full = s.sample(**{**kw, "preserve_mask": np.ones_like(mask)})
    assert np.array_equal(full, np.broadcast_to(reference, full.shape))


def test_inpainting_empty_mask_is_identity(sampler):
    kw = dict(n=2, steps=4, seed=9)
    plain = sampler.sample(**kw)
    edited = sampler.sample(**kw, reference=np.ones((16, 16), np.float32),
                            preserve_mask=np.zeros((16, 16), bool))
    assert np.array_equal(plain, edited)
    with pytest.raises(ValueError):
        sampler.sample(**kw, reference=np.ones((16, 16)))
    with pytest.raises(ValueError):
        sampler.sample(**kw, reference=np.ones((8, 8)), preserve_mask=np.ones((8, 8)))


def test_sampling_benchmark_measures_current_sampler(sampler):
    from nullscape.eval.performance import benchmark_sampler

    report = benchmark_sampler(sampler, batches=[1, 2], repeats=2, warmup=1, sampling={"steps": 3})
    assert report["device"] == "cpu" and len(report["rows"]) == 2
    for row in report["rows"]:
        assert row["steps"] == 3 and row["spacing"] == "quadratic"
        assert row["maps_per_second"] > 0 and len(row["batch_seconds"]) == 2
        assert row["latency_seconds_per_batch"] > 0 and not row["oom"]
    with pytest.raises(ValueError):
        benchmark_sampler(sampler, batches=[0], repeats=1)
