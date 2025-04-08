import torch
import torch.nn as nn
from typing import Tuple

class SiglipVisionConfig:
    def __init__(self,hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
                 num_attention_heads=12,num_channels=3, image_size=224,
                 patch_size=16,layer_norm_eps=1e-6,attention_dropout=0.0,
                 num_image_tokens: int = None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    # convert a raw image into embeddings
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid" # no padding
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches 
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", 
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)
        
    def forward(self, pixel_values) ->Tuple:
        batch_size = pixel_values.shape[0]
        _,_,height,width = pixel_values.shape # [batch_size, channels, height, width]
        patch_embeds = self.patch_embedding(pixel_values)
        # output of convolution is a 2x2 grid of patches
        # flatten to a single list
        embedding = patch_embeds.flatten(2)
        # [batch_size,  embed_dim,num_patchs,] ->[batch_size, num_patchs, embed_dim] 
        embedding = embedding.transpose(1, 2)
        # add position embedding
        # pos_embedding[0] is always added to first patch, etc...
        # we want the network to learn the spatial position of the patch
        embeddings = embedding + self.position_embedding(self.position_ids)
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        return hidden_states

class SiglipEncoder():
    # list of transformer layers
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    
    def forward(self, hidden_states):
        # save residual connection
        # [batch_size, num_patches, embed_dim]
        residual = hidden_states
        
        # normalize values
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # [batch_size, num_patches, embed_dim]
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values) ->Tuple:
        # [batch_size, channels, height,width] -> [batch_size, number_patches,embed_dim]
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(input_embeds=hidden_states)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_transformer = SiglipVisionTransformer(config)

    def forward(self, pixel_values) ->Tuple:
        # [batch_size, channels, height,width] -> [batch_size, number_patches,embed_dim]
        return self.vision_transformer(pixel_values)