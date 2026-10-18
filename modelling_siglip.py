from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size         = hidden_size
        self.intermediate_size   = intermediate_size
        self.num_hidden_layers   = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels        = num_channels
        self.image_size          = image_size
        self.patch_size          = patch_size
        self.layer_norm_eps      = layer_norm_eps
        self.attention_dropout   = attention_dropout
        self.num_image_tokens    = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        
        self.embed_dim  = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2 # across height and width
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(num_embeddings=self.num_positions, embedding_dim=self.embed_dim)
        
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).view(1, -1),
            persistent=False,
        )
    
    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape #[Batch_size, Channels, Height, Width]
        
        patch_embds = self.patch_embeddings(pixel_values) # [Batch_size, Channels, Height, Width] -> [Batch_size, Embed_dim, num_patches(h), num_patches(w)]
        
        embeddings = patch_embds.flatten(2) # [Batch_size, Embed_dim, num_patches_h, num_patches_w] -> [Batch_size, Embed_dim, num_patches]
        
        embeddings = embeddings.transpose(1, 2) # [Batch_size, Embed_dim, num_patches] -> [Batch_size, num_patches, Embed_dim]
        
        embeddings = embeddings + self.position_embeddings(self.position_ids) # [Batch_size, num_patches, Embed_dim]
        
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size) 
        self.fc2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states) # [Batch_size, Num_patches, Emb_dim] -> [Batch_size, Num_patches, Int_dim]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states) # [Batch_size, Num_patches, int_dim] -> [Batch_size, Num_patches, Emb_dim]
        
        return hidden_states

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = (self.head_dim)**-0.5
        self.dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size() # [Batch_size, Num_patches, Emb_dim]
        
        query_states = self.q_proj(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        key_states   = self.k_proj(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        value_states = self.v_proj(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_size, Num_heads, Num_patches, Head_dim]
        key_states   = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_size, Num_heads, Num_patches, Head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_size, Num_heads, Num_patches, Head_dim]
        
        # attn_wts = concat(softmax(Q @ K.T / sqrt(d_head)) @ V, dim=-1) @ O
        
        attn_wts = (query_states @ key_states.transpose(2,3)) * self.scale # [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_wts = nn.functional.softmax(attn_wts, dim=-1, dtype=torch.float32).to(query_states.dtype) # [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_wts = nn.functional.dropout(attn_wts, p=self.dropout, training=self.training) # [Batch_size, Num_heads, Num_patches, Num_patches]
        
        attn_op = attn_wts @ value_states # [Batch_size, Num_heads, Num_patches, Head_dim]
        
        attn_op = attn_op.transpose(1,2)  # [Batch_size, Num_patches, Num_heads, Head_dim]
        attn_op = attn_op.view(batch_size, seq_len, -1) # [Batch_size, Num_patches, Emb_dim]
        
        attn_op = self.out_proj(attn_op) # [Batch_size, Num_patches, Emb_dim]
        
        return attn_op, attn_wts      

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states):
        residual = hidden_states # [Batch_size, Num_patches, Emb_dim]
        
        hidden_states = self.layer_norm1(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        hidden_states, _ = self.self_attn(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        
        hidden_states = residual + hidden_states # [Batch_size, Num_patches, Emb_dim]
        
        residual = hidden_states # [Batch_size, Num_patches, Emb_dim]
        
        hidden_states = self.layer_norm2(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        hidden_states = self.mlp(hidden_states) # [Batch_size, Num_patches, Emb_dim]
        
        hidden_states = residual + hidden_states # [Batch_size, Num_patches, Emb_dim]
        
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, input_embs):
        hidden_states = input_embs
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim   = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder    = SiglipEncoder(config)
        
        self.post_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values):
        #  pixel_values: [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Emb_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layer_norm(last_hidden_state)
        
        return last_hidden_state
    
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values):
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Emb_dim]
        return self.vision_model(pixel_values=pixel_values)
    

        