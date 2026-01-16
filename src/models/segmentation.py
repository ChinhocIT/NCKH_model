import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SETRSegDecoder(nn.Module):
    """
    SETR-based Segmentation Decoder
    Uses DINOv2 ViT features for semantic segmentation.
    
    Default dim is 768 for DINOv2-B. Set to 1024 for ViT-L or 1536 for Qwen.
    """
    
    def __init__(self, 
                 vit_features_dim=768,  # DINOv2-B default
                 num_classes=172,
                 decoder_type='naive',  # 'naive', 'pup', 'mla'
                 device="cuda",
                 debug=False):
        super().__init__()
        
        self.device = device
        self.debug = debug
        self.vit_features_dim = vit_features_dim
        
        # Feature adapter
        self.feature_adapter = nn.Sequential(
            nn.Linear(vit_features_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # SETR Decoder variants
        if decoder_type == 'naive':
            # SETR-Naive: Simple upsampling
            self.decoder = self._build_naive_decoder(num_classes)
        elif decoder_type == 'pup':
            # SETR-PUP: Progressive UPsampling
            self.decoder = self._build_pup_decoder(num_classes)
        elif decoder_type == 'mla':
            # SETR-MLA: Multi-Level Aggregation
            self.decoder = self._build_mla_decoder(num_classes)
        
        print(f"   🎨 SETR-{decoder_type.upper()} decoder initialized")
    
    def _build_naive_decoder(self, num_classes):
        """SETR-Naive: 1x1 conv → Upsample"""
        return nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def _build_pup_decoder(self, num_classes):
        """SETR-PUP: Progressive upsampling"""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def _build_mla_decoder(self, num_classes):
        """SETR-MLA: Multi-level aggregation (simplified)"""
        self.aux_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        return nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, vit_features, target_size=None):
        """
        Args:
            vit_features: (B, N, C) from DINOv2 ViT (1024-dim for ViT-L)
            target_size: (H, W) output size
        
        Returns:
            seg_map: (B, num_classes, H, W)
        """
        B, N, C = vit_features.shape
        
        # Handle dimension mismatch (giống code gốc)
        if C != self.vit_features_dim:
            if not hasattr(self, 'dynamic_proj') or self.dynamic_proj is None:
                self.dynamic_proj = nn.Linear(C, self.vit_features_dim).to(vit_features.device)
            vit_features = self.dynamic_proj(vit_features)
        
        # Ensure float32
        if vit_features.dtype == torch.float16:
            vit_features = vit_features.float()
        
        # Adapt features
        adapted = self.feature_adapter(vit_features)  # (B, N, 768)
        
        # Reshape to 2D grid
        H_f = W_f = int(np.sqrt(N))
        if H_f * W_f != N:
            for h in range(int(np.sqrt(N)), 0, -1):
                if N % h == 0:
                    H_f = h
                    W_f = N // h
                    break
        
        adapted_2d = adapted.permute(0, 2, 1).reshape(B, 768, H_f, W_f)
        
        # Decode to segmentation map
        seg_map = self.decoder(adapted_2d)
        
        # Resize if needed
        if target_size is not None:
            seg_map = F.interpolate(
                seg_map, size=target_size,
                mode='bilinear', align_corners=True
            )
        
        return seg_map