# Testing Guide for automark3 Command

## Summary

This document provides testing instructions for the new `mel rotomap automark3` command that uses DINOv3 for improved mole matching performance.

## Environment Limitations

The automated CI environment has network restrictions preventing model downloads:
- **DINOv2**: 403 Forbidden when accessing `dl.fbaipublicfiles.com`
- **DINOv3**: 403 Forbidden when accessing `huggingface.co`

Therefore, **manual testing with unrestricted internet access is required** to validate performance improvements.

## Partial Test Results (Without Model Downloads)

### Test Setup
Using mel-datasets v0.1.0, removed 3 canonical moles:
- `f0732022`: (2925, 4705)
- `b38e00c7`: (2403, 4768)
- `d631043a`: (2277, 4055)

### Old Approach Results (guess-missing only, DINOv2 couldn't download)
- ✅ Moles found: 3/3 (100%)
- ❌ Moles matched (<50px): 0/3 (0%)
- Average distance: 168.82 pixels
- Individual: [111.5px, 236.9px, 158.0px]

## Manual Testing Instructions

### Prerequisites
```bash
git clone https://github.com/aevri/mel.git
cd mel
git checkout claude/update-dino-latest-GJu3z
uv pip install -e '.[dev]'
wget https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz
tar -xzf v0.1.0.tar.gz
```

### Quick Comparison Test

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

### Expected Improvements

DINOv3 advantages over DINOv2:
- 6.7B parameter teacher (vs 1.1B)
- 1.7B training images (vs 142M)
- Built-in register tokens
- RoPE position embeddings
- Should yield **higher match rate** and **lower average distance**

### Key Metrics

1. **Moles found**: N/3 discovered
2. **Moles matched**: N/3 within 50px
3. **Average distance**: Mean error in pixels
4. **Max distance**: Worst-case error

## Additional Testing Options

```bash
# Different model sizes (better accuracy, slower)
--dino-size base    # 768 features
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

## Code Validation ✅

- ✅ Static analysis passed
- ✅ Import fixes applied (AutoModel)
- ✅ Dependencies updated (transformers 4.56.0)
- ✅ Tests added (smoke + benchmark)
- ✅ Committed to `claude/update-dino-latest-GJu3z`

## Ready for Testing

The implementation is complete. Manual testing will validate the expected DINOv3 performance improvements over the existing DINOv2-based approach.
