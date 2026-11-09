# Supported Embedding Models

## 1. OpenCLIP Models

### SigLIP (Sigmoid Loss for Language-Image Pre-training)
- **ViT-SO400M-14-SigLIP-384** 
  - Pretrained: `webli`
  - Best overall performance
  - 384x384 resolution
  
- **ViT-B-16-SigLIP-384**
  - Pretrained: `webli`
  - Good balance of speed and accuracy

### Standard CLIP
- **ViT-L-14**
  - Pretrained: `openai`, `laion2b_s32b_b82k`
  - Large model, high accuracy
  
- **ViT-B-32**, **ViT-B-16**
  - Pretrained: `openai`, `laion2b_s34b_b79k`
  - Faster inference, good for real-time

### LongCLIP (Extended context)
- **LongCLIP-L**
  - Extended context length for detailed descriptions
  
### DFN (Diffusion-based Feature Network)
- **DFN5B-CLIP-ViT-H-14-384**
  - State-of-the-art features
  - 384x384 resolution

## 2. BLIP-2 Models

### Salesforce BLIP-2
- **blip2-opt-2.7b**
  - OPT language model backend
  - Good for VQA and captioning
  
- **blip2-flan-t5-xl** 
  - Flan-T5 language model backend
  - Better instruction following

### Pooling Strategies
- `cls`: Use [CLS] token
- `mean`: Average pooling
- `last`: Last layer features

## 3. Perception Encoder (PE-Core)

### Custom Perception Encoders
- Specialized encoders for specific domains
- Configurable architecture
- Fine-tuned for Vietnamese content

## Model Selection Guide

### For General Use
- **Recommended**: ViT-SO400M-14-SigLIP-384
- **Fast Alternative**: ViT-B-16-SigLIP-384

### For Vietnamese Content
- PE-Core models (if available)
- Fine-tuned CLIP variants

### For Detailed Descriptions
- LongCLIP-L
- BLIP-2 with appropriate pooling

### For Production
- Balance: ViT-L-14 (OpenAI)
- Speed Priority: ViT-B-16
- Quality Priority: DFN5B-CLIP-ViT-H-14-384

## Performance Comparison

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| ViT-B-16 | Small | Fast | Good | Low |
| ViT-L-14 | Large | Medium | Excellent | Medium |
| SigLIP-SO400M | XLarge | Slow | Best | High |
| BLIP-2 | Large | Medium | Excellent | Medium |
| PE-Core | Medium | Fast | Domain-specific | Low |

## Configuration Examples

### SigLIP (Best Quality)
```python
config = {
    "name": "ViT-SO400M-14-SigLIP-384",
    "backend": "open_clip",
    "pretrained": "webli",
    "precision": "fp32",
    "batch_size": 16
}
```

### CLIP (Balanced)
```python
config = {
    "name": "ViT-L-14",
    "backend": "open_clip", 
    "pretrained": "openai",
    "precision": "amp",
    "batch_size": 32
}
```

### BLIP-2 (Vision-Language)
```python
config = {
    "name": "Salesforce/blip2-opt-2.7b",
    "backend": "blip2",
    "precision": "fp16",
    "batch_size": 8,
    "pooling": "mean"
}
```

### PE-Core (Custom)
```python
config = {
    "name": "pe-core-vit-b",
    "backend": "perception_encoder",
    "precision": "fp32",
    "batch_size": 32,
    "perception_encoder": {
        "checkpoint_path": "/path/to/checkpoint.pth"
    }
}
```
