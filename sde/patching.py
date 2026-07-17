from typing import List, Tuple
import torch
import torch.nn as nn

class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)

class ECGFMFeatureExtractor(nn.Module):
    """
    Authentic ECG-FM 1D CNN Feature Extractor for Waveform Patching.
    Based on the wav2vec 2.0 ConvFeatureExtraction architecture.
    """
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 12,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(
            n_in, n_out, k, stride,
            is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        nn.LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast()
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    # GroupNorm requires num_groups, num_channels. In fairseq, they used dim, dim for instance-like norm
                    nn.GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            dim, k, stride = cl
            self.conv_layers.append(
                block(
                    in_d, dim, k, stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
    
    def forward(self, x):
        """
        Args:
            x: [batch, T, in_d]
        Returns:
            out: [batch, T', dim_out]
        """
        # x is B x T x C. Conv1d needs B x C x T
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
            
        # Return as B x T' x C' for torchcde compatibility
        return x.transpose(1, 2)
