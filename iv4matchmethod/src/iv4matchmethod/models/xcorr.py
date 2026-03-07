from __future__ import annotations

import torch
import torch.nn.functional as F


def depthwise_xcorr(search: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _, _ = search.shape
    search_reshaped = search.reshape(1, batch_size * channels, search.shape[-2], search.shape[-1])
    kernels = template.reshape(batch_size * channels, 1, template.shape[-2], template.shape[-1])
    response = F.conv2d(search_reshaped, kernels, groups=batch_size * channels)
    return response.reshape(batch_size, channels, response.shape[-2], response.shape[-1])

