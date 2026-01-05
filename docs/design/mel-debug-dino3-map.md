# mel-debug dino3-map

## Purpose

Locate a known mole in an unmarked target image using DINOv3 features from a source image where the mole is already marked. This is a debug/exploration tool for evaluating DINOv3's potential for mole matching in the `mel rotomap automark3` feature.

## Command Specification

```
mel-debug dino3-map OUTPUT_JPG TARGET_JPG TARGET_UUID SRC_JPG
    [--dino-size {small,base,large}]
    [--image-size SIZE]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `OUTPUT_JPG` | Path for the output heatmap image |
| `TARGET_JPG` | Image where we want to locate the mole (no moles file used) |
| `TARGET_UUID` | UUID of the mole to find (must exist in SRC_JPG's moles file) |
| `SRC_JPG` | Image containing the marked mole (moles file required) |
| `--dino-size` | Model size: small (384d), base (768d), large (1024d). Default: base |
| `--image-size` | Scale images to fit this size. Must be divisible by 16. Default: 896 |

### Example

```bash
mel-debug dino3-map output.jpg session2/photo.jpg abc123def session1/photo.jpg
```

## Algorithm

1. Load moles from SRC_JPG, find mole with TARGET_UUID
2. Scale SRC_JPG to normalized size, adjust mole coordinates accordingly
3. Extract DINOv3 features from that mole's patch in scaled SRC_JPG
4. Load TARGET_JPG (no moles file loaded), scale to normalized size
5. Extract features for all patches in scaled TARGET_JPG
6. Compute cosine similarity between source mole feature and all target patches
7. Render similarity heatmap on scaled TARGET_JPG (red overlay, green cross at best match)
8. Save heatmap to OUTPUT_JPG

## DINOv3 Background

DINOv3 (Distillation with No Labels v3) is Facebook/Meta's self-supervised Vision Transformer released in August 2025.

### Key Specifications

- **Architecture**: Vision Transformer (ViT) with patch size 16
- **Training**: Self-supervised on 1.7 billion images
- **Model sizes**: Small (384d), Base (768d), Large (1024d), Giant (1536d)
- **Token structure**: [CLS] + [4 register tokens] + [patch tokens]
- **Position encoding**: RoPE (Rotary Position Embeddings) for variable resolution

### Differences from DINOv2

| Feature | DINOv2 | DINOv3 |
|---------|--------|--------|
| Patch size | 14 | 16 |
| Token structure | [CLS] + patches | [CLS] + 4 registers + patches |
| Training images | 142M | 1.7B |
| Position encoding | Learned | RoPE |
| Model loading | torch.hub | timm |

### Model Loading

Models are loaded via `timm.create_model(model_name, pretrained=True)`:

- `vit_small_patch16_dinov3.lvd1689m` (small, 384d)
- `vit_base_patch16_dinov3.lvd1689m` (base, 768d)
- `vit_large_patch16_dinov3.lvd1689m` (large, 1024d)

## Implementation

### Files

| File | Purpose |
|------|---------|
| `mel/lib/dinov3.py` | DINOv3 model loading, feature extraction, heatmap rendering |
| `mel/cmddebug/dino3map.py` | Command implementation |
| `mel/cmddebug/meldebug.py` | Command registration |

### Key Functions in dinov3.py

```python
def load_dinov3_model(dino_size="base")
    # Returns: (Dinov3Model, feature_dim)

def scale_image_to_fit(image_rgb, image_size)
    # Returns: (scaled_image, (scale_x, scale_y))

def extract_mole_patch_feature(scaled_image, mole_x, mole_y, model)
    # Returns: Tensor [feature_dim]

def extract_all_patch_features(scaled_image, model)
    # Returns: Tensor [num_patches, feature_dim]

def compute_cosine_similarities(mole_feature, all_patch_features)
    # Returns: Tensor [num_patches]

def render_heatmap(image_rgb, similarities, image_height, image_width)
    # Returns: BGR image with red heatmap overlay
```

### Image Scaling

- Images are scaled to fit within `--image-size` (default 896px)
- Aspect ratio is preserved
- Dimensions are rounded down to be divisible by 16 (patch size)
- Mole coordinates are adjusted by the scale factor

### Heatmap Visualization

- Red channel intensity represents normalized similarity (0-1)
- Original image blended at 70% opacity
- Green cross marks the best match location (highest similarity patch)

## Dependencies

Added to `pyproject.toml`:
```toml
"timm>=1.0.20",
```

Uses [timm (PyTorch Image Models)](https://huggingface.co/timm) for model loading.

## Future Work

This debug command is a stepping stone toward `mel rotomap automark3`. Potential enhancements:

1. **Multiple source images**: Aggregate features from multiple views of the same mole
2. **Sliding window**: Process large images in overlapping windows for higher resolution
3. **Batch processing**: Process multiple moles in a single model forward pass
4. **Confidence thresholds**: Report match confidence and flag uncertain matches
5. **Integration with automark pipeline**: Use DINOv3 features for mole tracking across sessions
