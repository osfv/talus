# Engine export

`nullscape.export.engine.export_heightmap(h, world, out_dir, stem, formats, metadata, size)`
writes one normalized `[0, 1]` heightmap to engine-ready files plus a JSON
sidecar. CLI:

```bash
nullscape export --dataset base64 --index 123 --formats png16,r16,obj --out exports/
nullscape export --dataset base64 --index 123 --unity        # resample to next 2^n+1 first
```

## Formats

| Format | File | Encoding | Use |
|---|---|---|---|
| `png16` | `stem.png` | 16-bit grayscale PNG, `round(h*65535)` | Unreal, Godot, generic tools |
| `r16` | `stem.r16` | headerless uint16, **little-endian** | Unity Terrain "Import Raw" |
| `npy` | `stem.npy` | float32 in `[0, 1]` | Python tooling |
| `obj` | `stem.obj` | triangle mesh, meters, `f v//n` faces with vertex normals | quick preview in any DCC tool |

Heights map linearly: `height_m = h * max_height_m`. The minimum is 0, not sea
level; water is a rendering concept. The sidecar `stem.json` records:
- `resolution`
- `extent_m`: the world tile the map was generated for
- `footprint_m`: distance from the first to the last sample. Samples are cell
  centers, so this is one cell less than `extent_m`. Give this size to an engine.
- `cell_size_m`: sample spacing, `footprint_m / (resolution - 1)`
- `max_height_m`, `sea_level`, `sea_level_m`
- the encoding and byte order, and the files written
- the `metadata` dict (dataset, index, generator version, archetype) for provenance

`size=` resamples with bicubic interpolation before writing. Resampling is
corner-aligned, so `footprint_m` is preserved and only the spacing changes.
`unity_size(res)` gives the smallest valid Unity size (`2^n + 1`, at least 33)
that holds `res` samples: 64 -> 65, 128 -> 129. Sizes that are already valid
stay unchanged (65 -> 65, 513 -> 513).

## Unity

1. Export `r16`, optionally with `--unity` for a `2^n + 1` heightmap. Unity
   terrain requires 33, 65, 129, 257, 513, 1025, 2049 or 4097.
2. Select the Terrain asset → **Terrain Settings → Import Raw** (or right-click
   the terrain heightmap). Choose the `.r16` file.
3. In the import dialog: **Depth = 16 bit**, **Byte order = Windows**
   (little-endian), resolution = the sidecar's `resolution`.
4. Set the terrain size separately: **Width/Length = `footprint_m` m**,
   **Height = `max_height_m` m**. Heightmap value 0 sits at the terrain's
   base plane and 65535 at `height`. The mapping is already linear, so no
   offset is needed.

## Unreal Engine

UE Landscape accepts 16-bit grayscale PNG or little-endian RAW
(Landscape mode → New → **Import from File**). UE treats the 16-bit range as
-256 m .. +255.99 m times Z scale, quantized to `value/128 * Zscale` cm, with
the midpoint (32768) at the actor's Z.

- **Z scale** = `max_height_m * 100 / 512` (e.g. 1200 m -> 234.375). This maps
  the full `[0, 1]` range to `-max_height_m/2 .. +max_height_m/2` around the
  actor — raise the landscape actor by `max_height_m/2` if the lowest point
  should sit at Z = 0.
- **X/Y scale** = `cell_size_m * 100` (each landscape quad is 1 m at scale 100;
  e.g. 64 m cells -> 6400).
- Recommended heightmap sizes are `2^n * sections + 1` (505, 1009, 2017, 4033,
  8129). Export `--size 513` (or resample offline) to fit UE's component grid;
  a non-conforming size imports with edge padding.

## Godot

Both common terrain plugins read 16-bit data:

- **Terrain3D**: import `stem.png` as a heightmap (Assets → Import → Heightmap).
  Set the terrain's world size to `footprint_m` and the height range to
  `0 .. max_height_m`.
- **HTerrain (hterrain)**: use the 16-bit PNG (or RAW) as the heightmap source.
  Set map dimensions to `footprint_m` and height to `max_height_m`.

In both cases heights are absolute meters from 0, matching the sidecar
encoding; place water/sea level visuals at `sea_level_m`.
