import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedVisionLanguageAdaptor(nn.Module):
    """
    🔧 FIXED: Auto-detect Qwen dimensions và load weights correctly
    """
    
    def __init__(self, vision_dim=1536, llm_dim=1536, num_query_tokens=64):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, llm_dim)
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # ⚡ FLEXIBLE vision projection - will be resized if needed
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, llm_dim)
        )
        
        self.dynamic_proj = None
        
        # Layer norms
        self.ln_vision = nn.LayerNorm(llm_dim)
        self.ln_query = nn.LayerNorm(llm_dim)
        self.ln_output = nn.LayerNorm(llm_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim * 4, llm_dim),
            nn.Dropout(0.1)
        )
    
    def load_qwen_weights_enhanced(self, qwen_visual):
        """
        🔥 ULTRA FIXED: Auto-detect real Qwen dimensions và adapt
        """
        print("\n🔄 Loading enhanced Qwen weights (auto-detect mode)...")
        loaded_components = []
        
        try:
            if not hasattr(qwen_visual, 'merger'):
                print("   ⚠️ No merger found - using random init")
                return False
            
            merger = qwen_visual.merger
            
            # ============================================================
            # STEP 1: DETECT REAL DIMENSIONS
            # ============================================================
            print("\n🔍 [Step 1] Detecting Qwen dimensions...")
            
            detected_input_dim = None
            detected_output_dim = None
            source_layer = None
            source_name = None
            
            # Strategy 1: Check mlp
            if hasattr(merger, 'mlp'):
                mlp = merger.mlp
                
                if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
                    first_layer = mlp[0]
                    if isinstance(first_layer, nn.Linear):
                        detected_input_dim = first_layer.in_features
                        detected_output_dim = first_layer.out_features
                        source_layer = first_layer
                        source_name = "mlp[0]"
                        print(f"   ✅ Found in mlp[0]: {detected_input_dim} → {detected_output_dim}")
                
                elif isinstance(mlp, nn.Linear):
                    detected_input_dim = mlp.in_features
                    detected_output_dim = mlp.out_features
                    source_layer = mlp
                    source_name = "mlp"
                    print(f"   ✅ Found in mlp: {detected_input_dim} → {detected_output_dim}")
            
            # Strategy 2: Check down_proj
            if detected_input_dim is None and hasattr(merger, 'down_proj'):
                if isinstance(merger.down_proj, nn.Linear):
                    detected_input_dim = merger.down_proj.in_features
                    detected_output_dim = merger.down_proj.out_features
                    source_layer = merger.down_proj
                    source_name = "down_proj"
                    print(f"   ✅ Found in down_proj: {detected_input_dim} → {detected_output_dim}")
            
            # Strategy 3: Check proj
            if detected_input_dim is None and hasattr(merger, 'proj'):
                if isinstance(merger.proj, nn.Linear):
                    detected_input_dim = merger.proj.in_features
                    detected_output_dim = merger.proj.out_features
                    source_layer = merger.proj
                    source_name = "proj"
                    print(f"   ✅ Found in proj: {detected_input_dim} → {detected_output_dim}")
            
            if detected_input_dim is None:
                print("   ❌ Cannot detect Qwen projection dimensions")
                print("   📋 Available components:")
                for name, module in merger.named_children():
                    print(f"      - {name}: {type(module).__name__}")
                return False
            
            # ============================================================
            # STEP 2: ADAPT OUR PROJECTION LAYER
            # ============================================================
            print(f"\n🔧 [Step 2] Adapting vision_proj...")
            print(f"   Detected: {detected_input_dim} → {detected_output_dim}")
            print(f"   Current:  {self.vision_proj[0].in_features} → {self.vision_proj[0].out_features}")
            
            # Case 1: Perfect match - direct copy
            if (detected_input_dim == self.vision_proj[0].in_features and 
                detected_output_dim == self.vision_proj[0].out_features):
                
                print("   ✅ Perfect match! Copying weights directly...")
                self.vision_proj[0].weight.data.copy_(source_layer.weight.data)
                if source_layer.bias is not None:
                    self.vision_proj[0].bias.data.copy_(source_layer.bias.data)
                loaded_components.append(f"vision_proj (from {source_name})")
            
            # Case 2: Output matches, input different → Create adapter
            elif detected_output_dim == self.vision_proj[0].out_features:
                print(f"   🔄 Output matches ({detected_output_dim}), creating input adapter...")
                
                # Create adapter: detected_input_dim → vision_dim
                if detected_input_dim != self.vision_dim:
                    print(f"   📐 Adapter: {detected_input_dim} → {self.vision_dim}")
                    self.input_adapter = nn.Linear(
                        detected_input_dim, 
                        self.vision_dim
                    ).to(source_layer.weight.device)
                    
                    # Initialize with Xavier
                    nn.init.xavier_uniform_(self.input_adapter.weight)
                    if self.input_adapter.bias is not None:
                        self.input_adapter.bias.data.zero_()
                    
                    print("   ✅ Input adapter created")
                
                # Copy main projection
                self.vision_proj[0].weight.data.copy_(source_layer.weight.data)
                if source_layer.bias is not None:
                    self.vision_proj[0].bias.data.copy_(source_layer.bias.data)
                
                loaded_components.append(f"vision_proj+adapter (from {source_name})")
            
            # Case 3: Dimensions completely different → Interpolate
            else:
                print(f"   ⚠️ Dimension mismatch: {detected_input_dim}→{detected_output_dim} vs {self.vision_proj[0].in_features}→{self.vision_proj[0].out_features}")
                print("   🔄 Using interpolated initialization...")
                
                # Interpolate weights
                src_weight = source_layer.weight.data  # [out, in]
                tgt_in = self.vision_proj[0].in_features
                tgt_out = self.vision_proj[0].out_features
                
                # 🔥 FIX: Convert to float32 BEFORE interpolation (quantized weights are int8)
                if src_weight.dtype not in [torch.float32, torch.float16]:
                    print(f"   ⚠️ Converting {src_weight.dtype} → float32 for interpolation")
                    src_weight = src_weight.float()
                
                # Resize via interpolation
                src_weight_reshaped = src_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, out, in]
                tgt_weight = F.interpolate(
                    src_weight_reshaped,
                    size=(tgt_out, tgt_in),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [tgt_out, tgt_in]
                
                self.vision_proj[0].weight.data.copy_(tgt_weight)
                
                # Handle bias
                if source_layer.bias is not None and self.vision_proj[0].bias is not None:
                    src_bias = source_layer.bias.data
                    
                    # 🔥 FIX: Convert bias too
                    if src_bias.dtype not in [torch.float32, torch.float16]:
                        src_bias = src_bias.float()
                    
                    if detected_output_dim == tgt_out:
                        self.vision_proj[0].bias.data.copy_(src_bias)
                    else:
                        # Interpolate bias
                        tgt_bias = F.interpolate(
                            src_bias.unsqueeze(0).unsqueeze(0),
                            size=tgt_out,
                            mode='linear',
                            align_corners=False
                        ).squeeze()
                        self.vision_proj[0].bias.data.copy_(tgt_bias)
                
                loaded_components.append(f"vision_proj (interpolated from {source_name})")
                print("   ✅ Interpolated weights loaded")
            
            # ============================================================
            # STEP 3: Load LayerNorm (ALWAYS SAFE)
            # ============================================================
            print("\n🔧 [Step 3] Loading LayerNorm...")
            if hasattr(qwen_visual, 'ln_vision'):
                ln = qwen_visual.ln_vision
                if hasattr(ln, 'weight') and hasattr(ln, 'bias'):
                    if ln.weight.shape == self.ln_vision.weight.shape:
                        self.ln_vision.weight.data.copy_(ln.weight.data)
                        self.ln_vision.bias.data.copy_(ln.bias.data)
                        loaded_components.append("ln_vision")
                        print(f"   ✅ ln_vision loaded: {ln.weight.shape}")
            
            # ============================================================
            # STEP 4: Load FFN (WITH CHECKING)
            # ============================================================
            print("\n🔧 [Step 4] Loading FFN...")
            if hasattr(qwen_visual, 'ffn'):
                ffn = qwen_visual.ffn
                if isinstance(ffn, nn.Sequential):
                    # First layer
                    if len(ffn) > 0 and isinstance(ffn[0], nn.Linear):
                        if ffn[0].weight.shape == self.ffn[0].weight.shape:
                            self.ffn[0].weight.data.copy_(ffn[0].weight.data)
                            if ffn[0].bias is not None:
                                self.ffn[0].bias.data.copy_(ffn[0].bias.data)
                            loaded_components.append("ffn[0]")
                            print(f"   ✅ ffn[0] loaded: {ffn[0].weight.shape}")
                    
                    # Second layer (usually at index 3)
                    if len(ffn) > 3 and isinstance(ffn[3], nn.Linear):
                        if ffn[3].weight.shape == self.ffn[3].weight.shape:
                            self.ffn[3].weight.data.copy_(ffn[3].weight.data)
                            if ffn[3].bias is not None:
                                self.ffn[3].bias.data.copy_(ffn[3].bias.data)
                            loaded_components.append("ffn[3]")
                            print(f"   ✅ ffn[3] loaded: {ffn[3].weight.shape}")
            
            # ============================================================
            # SUMMARY
            # ============================================================
            print(f"\n{'='*60}")
            if loaded_components:
                print(f"✅ Successfully loaded: {', '.join(loaded_components)}")
                print(f"📊 Coverage: {len(loaded_components)}/4 components")
                return True
            else:
                print("⚠️ No components loaded - using random initialization")
                return False
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, vit_features, mask=None):
        """
        Forward pass with optional input adapter
        """
        B = vit_features.shape[0]
        
        if vit_features.dtype == torch.float16:
            vit_features = vit_features.float()
        
        C = vit_features.shape[-1]
        
        # Apply input adapter if exists (for dimension mismatch)
        if hasattr(self, 'input_adapter'):
            if C != self.vision_dim:
                vit_features = self.input_adapter(vit_features)
                C = self.vision_dim
        
        # Fallback dynamic projection
        if C != self.vision_dim:
            if not hasattr(self, 'dynamic_proj') or self.dynamic_proj is None or \
               self.dynamic_proj.in_features != C:
                self.dynamic_proj = nn.Linear(C, self.vision_dim).to(vit_features.device)
            vit_features = self.dynamic_proj(vit_features)
        
        # Main flow
        vision_projected = self.vision_proj(vit_features)
        vision_projected = self.ln_vision(vision_projected)
        
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        attended_queries, _ = self.cross_attention(
            query=query_tokens,
            key=vision_projected,
            value=vision_projected
        )
        
        query_tokens = self.ln_query(query_tokens + attended_queries)
        query_tokens = query_tokens + self.ffn(query_tokens)
        query_tokens = self.ln_output(query_tokens)
        
        return query_tokens