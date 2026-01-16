"""
VisionTextDecoder: Chuyển vision_tokens từ adaptor thành text description ngắn.
Được train trên VizWiz dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisionTextDecoder(nn.Module):
    """
    Decoder nhỏ chuyển vision_tokens (64, 1536) → text description.
    
    Architecture:
    - Cross-attention: text queries attend to vision tokens
    - Transformer decoder layers
    - Linear projection to vocab
    
    Note: Cần tokenizer từ Qwen để giữ consistency.
    """
    
    def __init__(
        self,
        vision_dim: int = 1536,
        hidden_dim: int = 512,
        vocab_size: int = 151936,  # Qwen2-VL vocab size
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Project vision tokens to decoder dim
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Token embedding (for decoder input)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection to vocab
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Special tokens (will be set from tokenizer)
        self.bos_token_id = 151643  # Default Qwen2 BOS
        self.eos_token_id = 151645  # Default Qwen2 EOS
        self.pad_token_id = 151643  # Default padding
        
        self._init_weights()
        
        print(f"  ✅ VisionTextDecoder initialized")
        print(f"     Vision dim: {vision_dim} → Hidden: {hidden_dim}")
        print(f"     Vocab size: {vocab_size}, Max length: {max_length}")
        print(f"     Decoder layers: {num_decoder_layers}, Heads: {num_heads}")
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def set_tokenizer(self, tokenizer):
        """Set special token IDs from tokenizer"""
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_token_id = tokenizer.bos_token_id
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        
        print(f"  ✅ Tokenizer set: BOS={self.bos_token_id}, EOS={self.eos_token_id}")
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for training.
        
        Args:
            vision_tokens: (B, num_vision_tokens, vision_dim) - from adaptor
            target_ids: (B, seq_len) - target token ids for teacher forcing
            target_mask: (B, seq_len) - attention mask for targets
            
        Returns:
            logits: (B, seq_len, vocab_size) - predictions
        """
        B = vision_tokens.shape[0]
        device = vision_tokens.device
        
        # Project vision tokens
        vision_memory = self.vision_proj(vision_tokens)  # (B, num_vision, hidden)
        
        if target_ids is None:
            # Inference mode - start with BOS
            target_ids = torch.full(
                (B, 1), 
                self.bos_token_id, 
                dtype=torch.long, 
                device=device
            )
        
        # Truncate if longer than max_length
        if target_ids.shape[1] > self.max_length:
            target_ids = target_ids[:, :self.max_length]
        
        # Clamp token IDs to valid range (CRITICAL: prevent CUDA assert)
        target_ids = target_ids.clamp(min=0, max=self.vocab_size - 1)
        
        seq_len = target_ids.shape[1]
        
        # Token + position embeddings
        token_embeds = self.token_embedding(target_ids)  # (B, seq_len, hidden)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.max_length - 1)  # Safety clamp
        pos_embeds = self.pos_embedding(positions)
        
        decoder_input = token_embeds + pos_embeds
        
        # Causal mask for decoder (prevent looking at future tokens)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Decode
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=vision_memory,
            tgt_mask=causal_mask
        )
        
        # Project to vocab
        logits = self.output_proj(decoder_output)  # (B, seq_len, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        vision_tokens: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Generate text from vision tokens.
        
        Args:
            vision_tokens: (B, num_vision_tokens, vision_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated_ids: (B, generated_length)
        """
        self.eval()
        
        if max_length is None:
            max_length = self.max_length
        
        B = vision_tokens.shape[0]
        device = vision_tokens.device
        
        # Project vision tokens
        vision_memory = self.vision_proj(vision_tokens)
        
        # Start with BOS token
        generated = torch.full(
            (B, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        for _ in range(max_length - 1):
            seq_len = generated.shape[1]
            
            # Token + position embeddings
            token_embeds = self.token_embedding(generated)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            pos_embeds = self.pos_embedding(positions)
            
            decoder_input = token_embeds + pos_embeds
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            
            # Decode
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=vision_memory,
                tgt_mask=causal_mask
            )
            
            # Get logits for last position only
            logits = self.output_proj(decoder_output[:, -1, :])  # (B, vocab_size)
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return generated
    
    def decode_to_text(
        self,
        vision_tokens: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Full pipeline: vision_tokens → text string
        
        Args:
            vision_tokens: (B, num_tokens, vision_dim)
            tokenizer: Qwen tokenizer
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            text: Decoded text string (for first item in batch)
        """
        # Generate token IDs
        generated_ids = self.generate(
            vision_tokens,
            max_length=max_length,
            temperature=temperature
        )
        
        # Decode to text
        text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        return text


class VisionTextDecoderLoss(nn.Module):
    """
    Loss function for training VisionTextDecoder.
    Cross-entropy with label smoothing and padding mask.
    """
    
    def __init__(self, pad_token_id: int = 151643, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: (B, seq_len, vocab_size) - model predictions
            target_ids: (B, seq_len) - target token ids
            
        Returns:
            loss: scalar tensor
        """
        # Shift for next-token prediction
        # logits: predict position 1, 2, 3, ...
        # targets: actual tokens at position 1, 2, 3, ...
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Reshape for cross-entropy
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
