# pcd-rs

## Overview

An rust project to provide basic operations for point cloud data, including reading, writing, voxelizing, downsampling, etc.

## development

```bash
uv venv
uv sync --dev
maturin develop --uv
```

please make sure you have `just` installed

```bash
sudo apt install just
```

then you can use `just` to run commands

```bash
just setup dev
```

## api docs

API's docs are hosted [here](https://hernandor.github.io/pcl-rustic/).
