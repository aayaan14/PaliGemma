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
        
        self.embed_dim = config.hidden_size
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
        
        embeddings = patch_embds.Flatten(2) # [Batch_size, Embed_dim, num_patches_h, num_patches_w] -> [Batch_size, Embed_dim, num_patches]
        
        embeddings = embeddings.transpose(1, 2) # [Batch_size, Embed_dim, num_patches] -> [Batch_size, num_patches, Embed_dim]
        
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        
        return embeddings

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
    

        