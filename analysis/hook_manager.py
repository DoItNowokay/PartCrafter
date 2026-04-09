import torch


class HookManager:

    def __init__(self, model):

        self.model = model
        self.handles = []
        self.attention_maps = []

    def _attention_hook(self, module, input, output):

        # Many transformers return (output, attn_probs)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            if isinstance(attn, torch.Tensor):
                self.attention_maps.append(attn.detach().cpu())

    def register_attention_hooks(self):

        for name, module in self.model.named_modules():

            if "attn" in name.lower():

                handle = module.register_forward_hook(
                    self._attention_hook
                )

                self.handles.append(handle)

    def clear(self):
        self.attention_maps = []

    def remove_hooks(self):

        for h in self.handles:
            h.remove()

        self.handles = []