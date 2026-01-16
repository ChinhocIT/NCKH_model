"""
DINOv2 ViT Encoder for SETR Segmentation Branch

This module provides a DINOv2 encoder for extracting dense features
for semantic segmentation, replacing Qwen ViT in the SETR branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT Encoder for dense feature extraction.
    
    Used for SETR segmentation branch - provides better dense features
    than Qwen ViT which is optimized for VLM tasks.
    
    Supported variants:
        - 'vits14': ViT-S/14, 21M params, dim=384
        - 'vitb14': ViT-B/14, 86M params, dim=768  
        - 'vitl14': ViT-L/14, 300M params, dim=1024 (recommended)
        - 'vitg14': ViT-g/14, 1.1B params, dim=1536
    """
    
    VARIANT_DIMS = {
        'vits14': 384,
        'vitb14': 768,
        'vitl14': 1024,
        'vitg14': 1536
    }
    
    def __init__(self, 
                 variant: str = 'vitb14',
                 device: str = 'cuda',
                 use_registers: bool = False):
        """
        Initialize DINOv2 encoder.
        
        Args:
            variant: Model variant ('vits14', 'vitb14', 'vitl14', 'vitg14')
            device: Device to load model on
            use_registers: Whether to use register variant (dinov2_vitl14_reg)
        """
        super().__init__()
        
        self.variant = variant
        self.device = device
        self.feature_dim = self.VARIANT_DIMS.get(variant, 1024)
        
        print(f"\n📦 Loading DINOv2 {variant.upper()}...")
        
        # Load model from torch hub
        model_name = f'dinov2_{variant}'
        if use_registers:
            model_name += '_reg'
            
        try:
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True
            )
        except Exception as e:
            print(f"⚠️ torch.hub failed: {e}")
            print("   Trying alternative loading...")
            # Fallback: try loading with trust_repo
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True,
                trust_repo=True
            )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get patch size (usually 14 for DINOv2)
        self.patch_size = 14
        
        print(f"✅ DINOv2 {variant.upper()} loaded")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Status: FROZEN ❄️")
    
    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOv2.
        
        Args:
            image_rgb: (H, W, 3) numpy array, uint8, RGB format
            
        Returns:
            Tensor (1, 3, H', W') normalized and resized
        """
        from PIL import Image
        import torchvision.transforms as T
        
        # Convert to PIL
        if isinstance(image_rgb, np.ndarray):
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        else:
            pil_image = image_rgb
            
        # DINOv2 standard preprocessing
        # Resize to multiple of patch_size
        h, w = pil_image.size[1], pil_image.size[0]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        
        if new_h == 0:
            new_h = self.patch_size
        if new_w == 0:
            new_w = self.patch_size
            
        transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)
    
    @torch.inference_mode()
    def extract_features(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Extract dense features from image.
        
        Args:
            image_rgb: (H, W, 3) numpy array, uint8, RGB format
            
        Returns:
            features: (1, N, C) where N = num_patches, C = feature_dim
        """
        # Preprocess
        pixel_values = self._preprocess_image(image_rgb)
        
        # Get features (exclude CLS token)
        # DINOv2 forward returns features including CLS token at position 0
        features = self.model.forward_features(pixel_values)
        
        # Handle different output formats
        if isinstance(features, dict):
            # Some versions return dict with 'x_norm_patchtokens'
            if 'x_norm_patchtokens' in features:
                patch_tokens = features['x_norm_patchtokens']
            elif 'x_prenorm' in features:
                patch_tokens = features['x_prenorm'][:, 1:]  # Remove CLS
            else:
                # Fallback to main output
                patch_tokens = features.get('x', features.get('last_hidden_state'))
                if patch_tokens is not None:
                    patch_tokens = patch_tokens[:, 1:]  # Remove CLS
        else:
            # Direct tensor output - remove CLS token
            patch_tokens = features[:, 1:]
            
        # Ensure float32
        if patch_tokens.dtype == torch.float16:
            patch_tokens = patch_tokens.float()
            
        return patch_tokens  # (1, N, C)
    
    @torch.inference_mode()
    def extract_features_with_cls(self, image_rgb: np.ndarray) -> tuple:
        """
        Extract features including CLS token.
        
        Returns:
            (cls_token, patch_tokens): CLS (1, C), patches (1, N, C)
        """
        pixel_values = self._preprocess_image(image_rgb)
        features = self.model.forward_features(pixel_values)
        
        if isinstance(features, dict):
            if 'x_norm_clstoken' in features:
                cls_token = features['x_norm_clstoken']
                patch_tokens = features.get('x_norm_patchtokens', features['x_prenorm'][:, 1:])
            else:
                all_tokens = features.get('x', features.get('last_hidden_state'))
                cls_token = all_tokens[:, 0:1]
                patch_tokens = all_tokens[:, 1:]
        else:
            cls_token = features[:, 0:1]
            patch_tokens = features[:, 1:]
            
        return cls_token.float(), patch_tokens.float()
    
    def get_feature_dim(self) -> int:
        """Return feature dimension for this variant."""
        return self.feature_dim
    
    def get_num_patches(self, image_size: tuple) -> int:
        """Calculate number of patches for given image size."""
        h, w = image_size
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        return h_patches * w_patches


# Convenience function for quick testing
def test_dinov2_encoder():
    """Quick test of DINOv2 encoder."""
    print("=" * 50)
    print("Testing DINOv2 Encoder")
    print("=" * 50)
    
    # Create encoder
    encoder = DINOv2Encoder(variant='vitb14', device='cuda')
    
    # Test with random image
    dummy_image = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    
    # Extract features
    features = encoder.extract_features(dummy_image)
    
    print(f"\nInput shape: {dummy_image.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {encoder.get_feature_dim()}")
    print(f"Expected patches: {encoder.get_num_patches((518, 518))}")
    
    # Verify shape
    B, N, C = features.shape
    assert B == 1, f"Expected batch=1, got {B}"
    assert C == 1024, f"Expected dim=1024, got {C}"
    
    print("\n✅ DINOv2 Encoder test PASSED!")
    return True


if __name__ == '__main__':
    test_dinov2_encoder()
