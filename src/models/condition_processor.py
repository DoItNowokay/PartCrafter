import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

class AttentionPooler(nn.Module):
    """
    Pools a sequence by cross-attending from a [CLS] token query
    to the rest of the sequence tokens (key/value).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # We use one LayerNorm for the [CLS] query
        self.query_norm = nn.LayerNorm(embed_dim)
        
        # And one LayerNorm for the sequence tokens (key/value)
        self.token_norm = nn.LayerNorm(embed_dim)
        
        # The cross-attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )
        # You could also add an MLP/FeedForward layer here
        # to make this a full Transformer block, but for pooling,
        # just the attention + residual is often enough.

    def forward(self, output_sequence: torch.Tensor) -> torch.Tensor:
        """
        Input sequence shape: (B, N+1, D)
        The [CLS] token is assumed to be at index 0.
        """
        
        # 1. Separate the [CLS] token (query) from the rest
        # We keep the sequence dim: (B, 1, D)
        cls_token = output_sequence[:, 0:1, :] 
        
        # The other tokens are the key/value: (B, N, D)
        other_tokens = output_sequence[:, 1:, :]
        
        # 2. Apply pre-normalization
        cls_norm = self.query_norm(cls_token)
        tokens_norm = self.token_norm(other_tokens)
        
        # 3. Perform cross-attention
        # Query = [CLS] token
        # Key/Value = All other tokens
        attn_output, _ = self.attn(
            query=cls_norm,
            key=tokens_norm,
            value=tokens_norm,
            need_weights=False
        )
        
        # 4. Add residual connection
        # The [CLS] token is updated by the info it gathered
        pooled_output = cls_token + attn_output
        
        # 5. Squeeze to remove the sequence dimension
        # (B, 1, D) -> (B, D)
        return pooled_output.squeeze(1)

class SharedTransformerBlock(nn.Module):
    """
    A standard Transformer Encoder block that will be shared
    by both image and text processing streams.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Shared Self-Attention
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_1 = nn.Dropout(dropout)
        
        # 2. Shared Feed-Forward Network
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Input sequence shape: (B, N, D)
        """
        # --- Self-Attention path ---
        # Pre-normalization (standard in modern transformers)
        x_norm = self.ln_1(sequence) 
        
        # Self-attention
        attn_output, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            need_weights=False
        )
        
        # First residual connection
        x = sequence + self.dropout_1(attn_output)
        
        # --- MLP path ---
        # Pre-normalization
        x_norm = self.ln_2(x)
        
        # MLP
        mlp_output = self.mlp(x_norm)
        
        # Second residual connection
        x_final = x + mlp_output
        
        return x_final


class ConditionProcessor(ModelMixin, ConfigMixin):
    """
    This is the main network you described.
    It takes encoded image/text sequences, prepends modality-specific
    [CLS] tokens, and processes them with a SharedTransformerBlock.
    
    Finally, it returns the processed [CLS] tokens as the
    pooled (B, 1024) embeddings.
    """
    @register_to_config
    def __init__(self,
            embed_dim: int = 1024,
            num_heads: int = 8,
            mlp_dim: int = 4096,
            projection_dim: int = 512,
            text_conditioning: str = "none",
            shared_blocks: int = 8
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_conditioning = text_conditioning
        self.shared_blocks = int(shared_blocks)

        if text_conditioning == "none":
            return
        elif text_conditioning == "direct_text":
            self.text_feature_dim = 768  # Assuming input text features are of this dimension
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        elif text_conditioning == "contrastive_text":
            # 1. Learnable, modality-specific [CLS] tokens
            self.image_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.text_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            
            # 2. text_projector to match embed_dim if needed
            self.text_feature_dim = 768  # Assuming input text features are of this dimension
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )

            # 3. The Shared Transformer Encoder Blocks (stacked)
            # Create `shared_blocks` independent Transformer blocks and apply them sequentially.
            self.shared_encoders = nn.ModuleList([
                SharedTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim
                ) for _ in range(self.shared_blocks)
            ])

            # 4. The final pooling layer
            self.final_pooler = AttentionPooler(
                embed_dim=embed_dim,
                num_heads=num_heads
            )

            # --- NEW: Loss Components ---
            # Projection head layers are now defined directly in this class
            self.image_projection = nn.Sequential(
                nn.Linear(embed_dim, projection_dim, bias=False),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim, bias=False),
            )
            self.text_projection = nn.Sequential(
                nn.Linear(embed_dim, projection_dim, bias=False),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim, bias=False),
            )
            
            # Learnable temperature parameter
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            raise ValueError(f"Unknown text_conditioning: {text_conditioning}")


    def forward(self, image: torch.Tensor, text: torch.Tensor, num_parts: torch.Tensor):
        """
        image_features: (B, 255, 1024) - Your encoded image patches
        text_features:  (B, 77, 1024) - Your encoded text tokens
        """
        if self.text_conditioning == "none":
            return None, image
        elif self.text_conditioning == "direct_text":
            B = text.shape[0]
            num_tokens = text.shape[1]
            text = text.view(-1, self.text_feature_dim)
            text = self.text_proj(text)
            text = text.view(B, num_tokens, self.embed_dim)
            return None, text
        elif self.text_conditioning == "contrastive_text":
            batch_size = image.shape[0]
            
            # --- 1. Prepare Image Sequence ---
            img_cls = self.image_cls_token.expand(batch_size, -1, -1) 
            img_seq = torch.cat([img_cls, image], dim=1)
            
            # --- 2. Prepare Text Sequence ---
            text = self.text_proj(text)
            txt_cls = self.text_cls_token.expand(batch_size, -1, -1)
            txt_seq = torch.cat([txt_cls, text], dim=1)
            
            # --- 3. Process with Shared Encoder ---
            img_output_seq = img_seq
            txt_output_seq = txt_seq
            # Process with stacked shared transformer blocks
            for block in self.shared_encoders:
                img_output_seq = block(img_output_seq)
                txt_output_seq = block(txt_output_seq)
 
            # --- 4. *** NEW: Pool using the shared AttentionPooler *** ---
            img_pooled = self.final_pooler(img_output_seq) # (B, 1024)
            txt_pooled = self.final_pooler(txt_output_seq) # (B, 1024)

            # --- 5. Calculate Contrastive Loss ---
            image_ids = torch.arange(len(num_parts), device=image.device).repeat_interleave(num_parts)
            loss = self.compute_contrastive_loss(
                img_pooled, txt_pooled, image_ids, image_ids
            )
            
            return loss, txt_output_seq[:, 1:, :]
        else:
            raise ValueError(f"Unknown text_conditioning: {self.text_conditioning}")
    
    def compute_contrastive_loss(self, 
                                img_pooled: torch.Tensor, 
                                txt_pooled: torch.Tensor, 
                                image_ids: torch.Tensor, 
                                text_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates the multi-positive contrastive loss.
        
        Args:
            img_pooled: (B_img, 1024) - Output from the forward() pass
            txt_pooled: (B_txt, 1024) - Output from the forward() pass
            image_ids: (B_img) - Tensors of IDs (e.g., class labels, text hashes)
            text_ids: (B_txt) - Tensors of IDs
            
        Returns:
            alignment_loss: A single scalar loss value
        """
        
        # 1. Project pooled features into the loss-space
        # Apply projection and then normalize for cosine similarity
        img_proj = F.normalize(self.image_projection(img_pooled), p=2, dim=-1) # B, projection_dim
        txt_proj = F.normalize(self.text_projection(txt_pooled), p=2, dim=-1) # B, projection_dim
        
        # 2. Calculate Logits (scaled cosine similarity)
        # Clamp logit_scale to prevent training instability
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # logits shape: (B_img, B_txt)
        logits_per_image = torch.matmul(img_proj, txt_proj.t()) * logit_scale
        # logits shape: (B_txt, B_img)
        logits_per_text = logits_per_image.t()

        # 3. Create Dense Ground-Truth Labels
        device = img_pooled.device 
        
        # Ensure IDs are on the correct device and are tensors
        if not isinstance(image_ids, torch.Tensor):
            image_ids = torch.tensor(image_ids, device=device)
        if not isinstance(text_ids, torch.Tensor):
            text_ids = torch.tensor(text_ids, device=device)
            
        # Use broadcasting to create the (B_img, B_txt) matrix of positive pairs
        # target_labels[i, j] = 1.0 if image_ids[i] == text_ids[j] else 0.0
        target_labels = (image_ids.unsqueeze(1) == text_ids.unsqueeze(0)).float()

        # 4. Calculate Symmetrical Cross-Entropy Loss
        
        # Image-to-Text Loss
        target_img = F.normalize(target_labels, p=1, dim=1, eps=1e-6)
        loss_img = F.cross_entropy(logits_per_image, target_img)
        
        # Text-to-Image Loss
        # Normalize the transposed target_labels by its rows (i.e., columns of original)
        target_txt = F.normalize(target_labels.t(), p=1, dim=1, eps=1e-6)
        loss_txt = F.cross_entropy(logits_per_text, target_txt)
        # Final loss is the average
        alignment_loss = (loss_img + loss_txt) / 2.0
        
        return alignment_loss

# class ConditionProcessor(ModelMixin, nn.Module):
#     # config_name = "ConditionProcessorConfig"
#     # @register_to_config
#     def __init__(self):
#         super().__init__()
#         self.text_feature_dim = 768
#         self.proj_dim = 1024

#         self.text_proj = nn.Sequential(
#             nn.Linear(self.text_feature_dim, self.proj_dim),
#             nn.ReLU(),
#             nn.Linear(self.proj_dim, self.proj_dim)
#         )
        

#     def forward(
#         self,
#         text: torch.Tensor = None,
#         image: torch.Tensor = None
#     ) -> torch.Tensor:
        
#         if text is not None:
#             B = text.shape[0]
#             text = text.view(-1, self.text_feature_dim)
#             text = self.text_proj(text)
#             text = text.view(B, -1, self.proj_dim)
#             return text
#         if image is not None:
#             return image
#         return None