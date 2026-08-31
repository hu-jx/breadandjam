import torch.nn as nn
class SafeAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        # Check if the tensor is on MPS
        if input.device.type == "mps":
            return super().forward(input.cpu()).to("mps")
        return super().forward(input)
