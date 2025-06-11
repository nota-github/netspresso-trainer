# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) activation function implementation.
    Also known as Swish activation function.
    
    Formula: SiLU(x) = x * sigmoid(x)
    """

    def __init__(self, inplace: bool = False):
        """
        Args:
            inplace (bool): If True, will do this operation in-place. Default: False
        """
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SiLU activation function.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying SiLU activation
        """
        # Step 1: Compute sigmoid of input
        sigmoid_x = torch.sigmoid(x)

        # Step 2: Element-wise multiplication of input and sigmoid
        if self.inplace:
            x.mul_(sigmoid_x)
            return x
        else:
            return x * sigmoid_x
