# Building the Documentation

This directory contains the Sphinx documentation for pcl-rustic.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Building HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `docs/build/html/`.

## Viewing the Documentation

Open `docs/build/html/index.html` in your web browser.

## Cleaning Build Files

```bash
cd docs
make clean
```

## Documentation Structure

- `source/` - RST source files
  - `index.rst` - Main documentation page
  - `api/` - API reference documentation
  - `examples/` - Usage examples
  - `conf.py` - Sphinx configuration
