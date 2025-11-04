import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# class ConditionProcessor(ConfigMixin, ModelMixin, nn.Module):
# class ConditionProcessor(nn.Module, ConfigMixin):
class ConditionProcessor(ModelMixin, nn.Module):
    # config_name = "ConditionProcessorConfig"
    # @register_to_config
    def __init__(self, config):
        super().__init__()
        self.text_feature_dim = 768
        self.proj_dim = 1024

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        

    def forward(
        self,
        text: torch.Tensor = None,
        image: torch.Tensor = None
    ) -> torch.Tensor:
        
        if text is not None:
            B = text.shape[0]
            text = text.view(-1, self.text_feature_dim)
            text = self.text_proj(text)
            text = text.view(B, -1, self.proj_dim)
            return text
        if image is not None:
            return image
        return None