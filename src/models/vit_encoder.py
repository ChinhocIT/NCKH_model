import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedFeatureExtractor(nn.Module):
    """
    FIXED: Focus extraction với attention mechanism
    """
    
    def __init__(self, feature_dim=1536):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Multi-head attention for better focus
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced pooling
        self.pool_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Learnable query for attention
        self.focus_query = nn.Parameter(
            torch.randn(1, 1, feature_dim)
        )
        
        self.dynamic_proj = None
        
        print(f"  ✅ Enhanced MaskedFeatureExtractor (dim: {feature_dim})")
    
    def forward(self, vit_features, mask):
        """
        Args:
            vit_features: (B, N, C)
            mask: (H, W) binary mask
        
        Returns:
            focused_features: (B, 1, C) - HIGHLY focused on masked region
        """
        B, N, C = vit_features.shape
        
        # Handle dimension mismatch
        if C != self.feature_dim:
            if not hasattr(self, 'dynamic_proj') or self.dynamic_proj is None or \
               self.dynamic_proj.in_features != C:
                self.dynamic_proj = nn.Linear(C, self.feature_dim).to(vit_features.device)
            
            vit_features = self.dynamic_proj(vit_features)
            C = self.feature_dim
        
        # Tìm grid size
        H_f = W_f = None
        for h in range(int(np.sqrt(N)), 0, -1):
            if N % h == 0:
                H_f = h
                W_f = N // h
                break
        
        if H_f is None or W_f is None:
            H_f = W_f = int(np.sqrt(N))
        
        # Resize mask to feature map
        mask_tensor = torch.from_numpy(mask).float()
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(H_f, W_f),
            mode='nearest'
        ).to(vit_features.device)
        
        mask_flat = mask_resized.flatten(1).expand(B, -1)  # (B, N)
        
        # === IMPROVED: Strong focus on masked region ===
        
        # 1. Soft weight (không zero hoàn toàn)
        mask_weights = torch.sigmoid(
            (mask_flat - 0.5) * 10  # Sharp transition nhưng không zero
        ).unsqueeze(-1)  # (B, N, 1)

        # 2. Weighted masking
        masked_features = vit_features * mask_weights

        # 3. Dynamic boost based on mask coverage (smaller mask = more boost)
        mask_ratio = mask_weights.sum() / mask_weights.numel()
        boost_factor = 1.0 + (1.0 - mask_ratio.clamp(0.1, 0.9)) * 2.0  # Range: 1.2x - 2.8x
        masked_features = masked_features * boost_factor

        # 4. Create attention mask (True = ignore, False = attend)
        attn_mask = (mask_flat < 0.5)  # (B, N)
        
        # 5. Get focus query BEFORE using it
        focus_query = self.focus_query.expand(B, -1, -1)  # (B, 1, C)
        
        # 6. Apply attention with weighted features
        attended_features, attn_weights = self.attention(
            query=focus_query,
            key=vit_features,  # Use original as key
            value=masked_features,  # Use weighted as value
            key_padding_mask=attn_mask
        )
        
        # 7. Project
        focused = self.pool_proj(attended_features)  # (B, 1, C)
        
        return focused