# UrbanOmniDetect + I24-3D evaluation container

This container packages the evaluation adapter with UrbanOmniDetect and its
Python dependencies.

## Important data-format assumption

The current adapter expects this cache layout:

```text
i24-cache/
  im/scene_1/p1c1/0000.png
  label/scene_1/p1c1/0000.cpkl
```

Each `.cpkl` file must deserialize to a list of objects with an `im_box`
field containing eight projected 2D corners. Verify this matches your actual
I24-3D export before running. If your annotations are instead state records
(`x, y, l, w, h, direction`), the loader must use I24-3D's official state-to-
image projection utility; the container cannot infer camera geometry.

## Build

Docker with BuildKit is recommended. Building downloads UrbanOmniDetect from
GitHub and installs its requirements, so internet access is required.

```bash
cp .env.example .env
# Edit .env paths, checkpoint filename, scene and camera.
mkdir -p outputs
docker compose build
```

## Run

```bash
docker compose run --rm i24-eval
```

Quick smoke test with a few frames:

```bash
docker run --rm \
  -v "$PWD/i24-cache:/data/i24-cache:ro" \
  -v "$PWD/checkpoints:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  urbanomnidetect-i24-eval:local \
  --cache /data/i24-cache --scene 1 --camera p1c1 \
  --model /models/model.pt --uod-version v1 --max-images 10 \
  --device cpu --report-csv /outputs/smoke.csv
```

For GPU inference, install NVIDIA Container Toolkit on the host, set
`DEVICE=cuda:0` in `.env`, and run with GPU access, for example:
`docker compose run --rm -- gpus all i24-eval` is not valid Compose syntax;
instead add `gpus: all` under the service in `docker-compose.yml`, then run
`docker compose run --rm i24-eval`. The image includes CUDA libraries, but
GPU access must be enabled by the host runtime.

## Output and interpretation

The CSV contains AP2D, APBEV and AP3D at the requested IoU thresholds.
APBEV/AP3D are homography-based estimates, not direct 3D IoU against
independently measured metric XYZ corners. All objects are treated as one
vehicle class by the adapter.

## Troubleshooting

- `No .png images found`: check `im/scene_N/CAMERA` and `.env`.
- Missing paired `.cpkl`: ensure annotation and image basenames match.
- Missing `im_box`: your export differs from the expected format; adapt the
  loader or project the I24 state with the dataset's official utilities.
- Checkpoint loading errors: match `UOD_VERSION` to the checkpoint generation.
