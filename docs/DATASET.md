# Dataset format and generation pipeline

Procedural heightmap datasets are built by `nullscape gen-dataset` (see
`src/nullscape/data/build.py`). Everything below is derived from the resolved
config, which is stored verbatim in `manifest.json`.

## On-disk layout

`<data_root>/<name>/` where `data_root` is `$NULLSCAPE_DATA_ROOT` or `<repo>/data`:

| File | Content |
|---|---|
| `heights.npy` | uint16 `[N, R, R]`; `h = value / 65535`, memory-mapped on read |
| `conditions.npy` | float32 `[N, K]`, raw measured properties ordered as `manifest["condition_keys"]` |
| `labels.npy` | int64 `[N]` archetype ids |
| `meta.jsonl` | one JSON record per map: `index`, `seed`, `archetype`, `archetype_id`, `params`, `secondary*` blend info, `clipped_fraction`, `metrics`, `traversability` |
| `split_{train,val,test}.npy` | int64 index arrays, disjoint and covering `0..N-1` |
| `manifest.json` | provenance: resolved config + `config_sha256`, `generator_version`, world spec, split sizes, `condition_stats` (mean/std on the train split only), archetype counts, clipped-fraction and traversability-pass summaries, git commit/dirty, versions, wall time |

`TerrainStore` (storage.py) reads a dataset; `TerrainDataset` (dataset.py)
serves model-ready samples: `x` float32 `[1, R, R]` in `[-1, 1]` (`h*2-1`),
`cond` z-scored with the manifest train-split stats, `label`, `index`, plus
optional dihedral augmentation on train.

## Generation pipeline

`generate(index, global_seed, world, cfg)` in `terrain/generator.py`:

1. **Archetype synthesis.** One of six archetypes is drawn from
   `archetype_weights` (label ids are a public contract:
   `0 plains, 1 hills, 2 mountains, 3 ridges, 4 islands, 5 mesas`). Parameters
   are sampled per map, and the base field is evaluated on a `supersample`×
   (default 2×) grid in world-kilometre coordinates, so the same parameters
   describe the same terrain at any resolution. fBm / billow / ridged
   multifractal noise with optional domain warping; octaves above the grid
   Nyquist are skipped.
2. **Optional blend.** With probability `blend_probability` (default 0.25) a
   second archetype's field is blended through a low-frequency smoothstepped
   fBm mask. The label stays the primary archetype; the secondary archetype
   and mean blend weight are recorded in `meta.jsonl`.
3. **Area downsample** to the target resolution and conversion to meters.
4. **Stream-power erosion.** D8 steepest-descent routing; drainage area from a
   sparse linear solve; explicit incision `dz = -k·A^m·S`, capped per step for
   stability; cells at or below sea level do not erode. Iterations/`k` come
   from the archetype params.
5. **Hillslope diffusion.** Explicit linear diffusion (mass-conserving,
   reflecting boundaries).
6. **Thermal erosion.** Talus-angle relaxation, mass-conserving
   (`thermal_iters` from config, `talus_deg` from archetype params).

## Normalization

Output heights are `meters / world.max_height_m`, clipped to `[0, 1]` — a
**global** normalization shared by every map. Per-map min-max normalization is
deliberately not done: it would stretch flat plains to full height and destroy
the amplitude information the model should learn (and that the conditioning
vector reports). Clipping is rare by construction; the fraction of clipped
cells is recorded per map in `meta.jsonl` and summarized in `manifest.json`
(`summary.clipped_fraction_*`).

## Reproducibility

Each map's RNG is `np.random.SeedSequence([global_seed, index])`, so the output
is a pure function of (seed, index, world, generator config) — identical
regardless of worker count or completion order, and bit-identical across runs.
Bump `GENERATOR_VERSION` in `generator.py` whenever output semantics change;
it is recorded in every manifest and meta record.

## Commands

```bash
nullscape gen-dataset --config configs/dataset/smoke.yaml              # 512 maps, 64x64
nullscape gen-dataset --config configs/dataset/base64.yaml --workers 8 # 50k maps
nullscape gen-dataset --config configs/dataset/base64.yaml \
    --set n=1000 --set seed=42 --set generator.blend_probability=0.4
nullscape viz --dataset smoke --split val --n 36 --out reports/smoke
```

Existing datasets are not overwritten unless `--overwrite` is passed; a
`.partial` directory is used while building and renamed on success, so an
interrupted build never looks complete.
