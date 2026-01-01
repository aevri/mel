# Testing Guide for automark3 Command

## Summary

This document provides testing instructions for the `mel rotomap automark3` command that uses DINOv2 for improved mole matching performance.

## Implementation Details

The automark3 command uses DINOv2 via torch.hub from the `facebookresearch/dinov2` repository. This is publicly accessible without authentication, unlike DINOv3 models on HuggingFace which are gated repositories.

## Prerequisites

```bash
git clone https://github.com/aevri/mel.git
cd mel
git checkout claude/update-dino-latest-GJu3z
uv pip install -e '.[dev]'
wget https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz
tar -xzf v0.1.0.tar.gz
```

## Quick Comparison Test

```bash
#!/bin/bash
BASE="mel-datasets-0.1.0/m1/rotomaps/parts/Trunk/Back"
SRC="$BASE/2025_06_12/0.jpg"
TGT="$BASE/2025_06_13/0.jpg"
JSON="$TGT.json"

# Backup and remove 3 moles
cp "$JSON" "$JSON.bak"
python3 -c "import json; d=json.load(open('$JSON.bak')); json.dump(d[3:],open('$JSON','w'))"

# Test old approach
mel rotomap guess-missing "$SRC" "$TGT"
mel rotomap guess-refine --dino-size small "$SRC" "$TGT"
# Measure accuracy here...

# Reset and test new approach
cp "$JSON.bak" "$JSON"
python3 -c "import json; d=json.load(open('$JSON.bak')); json.dump(d[3:],open('$JSON','w'))"
mel rotomap automark3 --reference "$SRC" --target "$TGT" --dino-size small
# Measure accuracy here...

# Restore
cp "$JSON.bak" "$JSON"
```

## Model Sizes

Available DINOv2 model sizes via torch.hub:
- `small`: 384 feature dimensions (fastest)
- `base`: 768 feature dimensions (default, good balance)
- `large`: 1024 feature dimensions (more accurate)
- `giant`: 1536 feature dimensions (most accurate, slowest)

## Key Metrics

1. **Moles found**: N/3 discovered
2. **Moles matched**: N/3 within 50px
3. **Average distance**: Mean error in pixels
4. **Max distance**: Worst-case error

## Additional Testing Options

```bash
# Different model sizes (better accuracy, slower)
--dino-size base    # 768 features (default)
--dino-size large   # 1024 features

# Multiple reference images
--reference img1.jpg img2.jpg img3.jpg

# Aggregation methods
--aggregation mean    # Average features (default)
--aggregation median  # Median features (more robust to outliers)
--aggregation first   # Use first occurrence only

# Debug visualization
--debug-images  # Save similarity heatmaps
```

## Code Validation

- Static analysis: `./meta/static_tests.sh`
- Unit tests: `./meta/unit_tests.sh`
- Full tests: `pytest --doctest-modules`

## Ready for Testing

The implementation is complete and uses DINOv2 via torch.hub which is publicly accessible. Tests should pass without authentication requirements.
