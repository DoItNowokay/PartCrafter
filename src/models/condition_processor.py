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

class LatentEditingBlock(nn.Module):
    """
    Modifies DINOv2 image embeddings based on CLIP text embeddings.
    This is effectively a Transformer Decoder layer.
    """
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Layer norms
        self.norm_image = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # The core of the module: Cross-Attention
        # The image embeddings (Q) "attend to" the text embeddings (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # Expects (B, N, D)
        )
        
        # Standard Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, image_embeds, text_embeds_projected):
        # image_embeds: (B, 257, 1024)
        # text_embeds_projected: (B, 77, 1024)
        
        # 1. Normalize the image embeddings (Query)
        image_normed = self.norm_image(image_embeds)
        
        # 2. Normalize the text embeddings (Key, Value)
        text_normed = self.norm_text(text_embeds_projected)
        
        # 3. Cross-Attention
        # Each of the 257 image tokens will "look at" the 77 text tokens
        # to find relevant editing information.
        attn_output, _ = self.cross_attn(
            query=image_normed,           # (B, 257, 1024)
            key=text_normed,              # (B, 77, 1024)
            value=text_normed             # (B, 77, 1024)
        )
        
        # 4. First Residual Connection (Add the text info)
        # This is the "delta" you were talking about
        x = image_embeds + attn_output
        
        # 5. Feed-Forward Network (to process the new info)
        ffn_output = self.ffn(self.norm_ffn(x))
        
        # 6. Second Residual Connection
        modified_image_embeds = x + ffn_output
        
        # output: (B, 257, 1024) - same shape as input!
        return modified_image_embeds


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
            inner_dim: int = 2048,
            num_heads: int = 8,
            mlp_dim: int = 4096,
            projection_dim: int = 512,
            text_conditioning: str = "none",
            editing: str = "none",
            shared_blocks: int = 8
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.text_conditioning = text_conditioning
        self.editing = editing
        self.shared_blocks = int(shared_blocks)
        self.text_feature_dim = 768  # Assuming input text features are of this dimension

        if text_conditioning == "none":
            pass
        elif text_conditioning == "direct_text":
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        elif text_conditioning == "direct_text_improved":
            self.text_tokens = 77
            # Learnable tokens that will form the new sequence
            self.learnable_queries = nn.Parameter(torch.randn(1, self.text_tokens, self.embed_dim))
            
            # Cross-attention layer
            # It will attend to the CLIP tokens (K, V)
            # using the learnable queries (Q)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, 
                kdim=self.text_feature_dim,  # Key dim is from CLIP
                vdim=self.text_feature_dim,  # Value dim is from CLIP
                num_heads=num_heads, 
                batch_first=True
            )
            
            # A standard transformer feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim)
            )
            self.norm1 = nn.LayerNorm(self.embed_dim)
            self.norm2 = nn.LayerNorm(self.embed_dim)
        elif text_conditioning == "adaln_text":
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.adaln_text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.inner_dim),
                nn.SiLU(),  # smoother than ReLU → less likely to explode in fp16
                nn.Linear(self.inner_dim, self.inner_dim)
            )

        elif text_conditioning == "contrastive_text":
            # 1. Learnable, modality-specific [CLS] tokens
            self.image_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.text_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            
            # 2. text_projector to match embed_dim if needed
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
        
        elif text_conditioning == "contrastive_text_pooled":
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.image_projection = lambda x: x
            self.text_projection = lambda x: x

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        elif text_conditioning == "contrastive_text_michelangelo":
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_feature_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            # self.image_projection = nn.Sequential(
            #     nn.Linear(embed_dim, projection_dim, bias=False),
            #     nn.ReLU(),
            #     nn.Linear(projection_dim, projection_dim, bias=False),
            # )
            # self.text_projection = nn.Sequential(
            #     nn.Linear(embed_dim, projection_dim, bias=False),
            #     nn.ReLU(),
            #     nn.Linear(projection_dim, projection_dim, bias=False),
            # )
            self.image_projection = lambda x: x
            self.text_projection = lambda x: x
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            raise ValueError(f"Unknown text_conditioning: {text_conditioning}")

        if editing == "none":
            pass
        elif editing == "direct_tweak_latent":
            self.editing_module = nn.ModuleList([])
            for _ in range(self.shared_blocks):
                self.editing_module.append(
                    LatentEditingBlock(
                        embed_dim=self.embed_dim,
                        num_heads=num_heads
                    )
                )
        elif editing == "l1_tweak_latent":
            self.editing_module = nn.ModuleList([])
            for _ in range(self.shared_blocks):
                self.editing_module.append(
                    LatentEditingBlock(
                        embed_dim=self.embed_dim,
                        num_heads=num_heads
                    )
                )
            
        elif editing == "text_cross_attn":
            pass
        elif editing == "source_cross_attn":
            pass
        else:
            raise ValueError(f"Unknown editing: {editing}")

        self.apply(self._init_weight)

    # --- [NEW] INITIALIZATION FUNCTION ---
    def _init_weight(self, m):
        """
        Applies Xavier uniform initialization to Linear layers and
        resets LayerNorm layers to default.
        """
        if isinstance(m, nn.Linear):
            # Apply Xavier uniform initialization to the weight
            torch.nn.init.xavier_uniform_(m.weight)
            # Initialize the bias to zero, if it exists
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LayerNorm):
            # Reset LayerNorm to its default init (weight=1, bias=0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(self,
                image: torch.Tensor,
                text: torch.Tensor,
                image_pooled: torch.Tensor,
                text_pooled: torch.Tensor,
                num_parts: torch.Tensor,
                target_image_embed: torch.Tensor = None
        ):
        """
        image_features: (B, 255, 1024) - Your encoded image patches
        text_features:  (B, 77, 1024) - Your encoded text tokens
        """
        loss = None
        combined_feats = None
        if self.text_conditioning == "none":
            loss = None
            combined_feats = text
        elif self.text_conditioning == "direct_text" and self.editing != "text_cross_attn":
            B = text.shape[0]
            num_tokens = text.shape[1]
            text = text.view(-1, self.text_feature_dim)
            text = self.text_proj(text)
            text = text.view(B, num_tokens, self.embed_dim)
            # return None, text
            loss = None
            combined_feats = text
        elif self.text_conditioning == "direct_text" and self.editing == "text_cross_attn":
            pass
        elif self.text_conditioning == "direct_text_improved":
            # text_token_embeddings shape: (B, 77, 768)
            
            # Expand learnable queries to match batch size
            # queries shape: (B, 32, 1024)
            queries = self.learnable_queries.repeat(text.size(0), 1, 1)

            # Attend to the text embeddings
            # Q = queries
            # K, V = text
            attn_output, _ = self.cross_attn(
                query=queries, 
                key=text, 
                value=text
            )
            
            # Add & Norm (like a transformer)
            x = self.norm1(queries + attn_output)
            
            # FFN
            ffn_output = self.ffn(x)
            
            # Add & Norm
            conditioning_signal = self.norm2(x + ffn_output)
            
            # output shape: (B, 32, 1024)
            # return None, conditioning_signal
            # text = conditioning_signal
            loss = None
            combined_feats = conditioning_signal
        elif self.text_conditioning == "adaln_text":
            # print(type(text))
            if isinstance(text, tuple):
                text_pooled = text[1]   # shape [B, 768]
                text = text[0]          # shape [B, num_tokens, 768]
                B = text.shape[0]
                num_tokens = text.shape[1]

                text = text.view(-1, self.text_feature_dim)
                text = self.text_proj(text)
                text = text.view(B, num_tokens, self.embed_dim)

                # project pooled text
                projected_text_pooled = self.adaln_text_proj(text_pooled)

                # return None, [text, projected_text_pooled]
                # text = [text, projected_text_pooled]
                loss = None
                combined_feats = [text, projected_text_pooled]
            else:
                B = text.shape[0]
                num_tokens = text.shape[1]
                text = text.view(-1, self.text_feature_dim)
                text = self.text_proj(text)
                text = text.view(B, num_tokens, self.embed_dim)
                # return None, text
                loss = None
                combined_feats = text

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
            # return loss, txt_output_seq[:, 1:, :]
            # text = txt_output_seq[:, 1:, :]
            loss = None
            combined_feats = txt_output_seq[:, 1:, :]
        
        elif self.text_conditioning == "contrastive_text_pooled":
            text = self.text_proj(text) # (B, N, 1024)
            txt_pooled = self.text_proj(text_pooled)  # (B, 1024)

            image_ids = torch.arange(len(num_parts), device=image.device).repeat_interleave(num_parts)
            loss = self.compute_contrastive_loss(
                image_pooled, txt_pooled, image_ids, image_ids
            )
            # return loss, text
            loss = None
            combined_feats = text

        elif self.text_conditioning == "contrastive_text_michelangelo":
            text = self.text_proj(text)
            txt_pooled = torch.mean(text, dim=1)  # (B, 1024)
            img_pooled = torch.mean(image, dim=1)  # (B, 1024)

            image_ids = torch.arange(len(num_parts), device=image.device).repeat_interleave(num_parts)
            loss = self.compute_contrastive_loss(
                img_pooled, txt_pooled, image_ids, image_ids
            )
            # return loss, text
            loss = None
            combined_feats = text
        else:
            raise ValueError(f"Unknown text_conditioning: {self.text_conditioning}")
        
        if self.editing == "none":
            pass
        elif self.editing == "direct_tweak_latent":
            # Apply each LatentEditingBlock sequentially
            for edit_block in self.editing_module:
                image = edit_block(image, text)
            combined_feats = image
        elif self.editing == "l1_tweak_latent":
            # Apply each LatentEditingBlock sequentially
            for edit_block in self.editing_module:
                image = edit_block(image, text)
            combined_feats = image
            # L1 loss is often more stable and robust than L2 (MSE) for latents
            l1_loss_latent = F.l1_loss(combined_feats, target_image_embed)
            if loss is not None:
                loss = loss + l1_loss_latent*0.25
            else:
                loss = l1_loss_latent*0.25
        elif self.editing == "text_cross_attn":
            loss = None
            combined_feats = [text, image]
        elif self.editing == "source_cross_attn":
            pass
        else:
            raise ValueError(f"Unknown editing: {self.editing}")

        return loss, combined_feats
    
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