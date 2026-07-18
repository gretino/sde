import torch
import torch.nn as nn
import torchcde
from sde.patching import ECGFMFeatureExtractor

class CDEFunc(nn.Module):
    """
    Vector field for the continuous-time context encoder.
    """
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = nn.Linear(hidden_channels, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, input_channels * hidden_channels)
        
        # Zero-initialize the vector field output to stabilize initial integration trajectories
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, t, z):
        # z: [batch, hidden_channels]
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return torch.tanh(z)

class PhysiologicalEncoder(nn.Module):
    """
    Converts past ECG waveform context into a latent initial state z_t0
    using ECG-FM patching and Neural CDE integration.
    """
    def __init__(self, in_leads: int, conv_layers: list, latent_dim: int):
        super().__init__()
        self.patcher = ECGFMFeatureExtractor(conv_layers=conv_layers, in_d=in_leads)
        
        patcher_out_dim = conv_layers[-1][0]
        cde_input_dim = patcher_out_dim + 1 # +1 for time channel
        
        self.cde_func = CDEFunc(cde_input_dim, latent_dim)
        self.initial_mapping = nn.Linear(cde_input_dim, latent_dim)
        
        # Near-zero initialization to silence initial latent noise and enable coordinate bootstrapping
        nn.init.normal_(self.initial_mapping.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.initial_mapping.bias, 0.0)
        
    def forward(self, waveform: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, T, leads]
            times: [T] context timestamps
        Returns:
            z_t0: [batch, latent_dim] Macro Physiological State at t=0
        """
        # 1. Waveform Patching
        patched_features = self.patcher(waveform) # [batch, T', patcher_out_dim]
        batch_size, T_prime, _ = patched_features.shape
        
        # 2. Downsample times to match patched features
        new_times = torch.linspace(times[0].item(), times[-1].item(), T_prime, device=waveform.device)
        
        # Add time channel to features
        time_channel = new_times.unsqueeze(0).unsqueeze(-1).expand(batch_size, T_prime, 1)
        cde_input = torch.cat([time_channel, patched_features], dim=-1)
        
        # 3. Build Neural CDE continuous path
        coeffs = torchcde.natural_cubic_coeffs(cde_input, new_times)
        X = torchcde.CubicSpline(coeffs, new_times)
        
        # 4. Integrate along path to the anchor time
        z0 = self.initial_mapping(X.evaluate(new_times[0]))
        z_t = torchcde.cdeint(X=X, z0=z0, func=self.cde_func, t=new_times, adjoint=False)
        
        # Final state corresponds to Normalized Anchor Time (t=0)
        z_t0 = z_t[:, -1, :]
        return z_t0

def get_interpolated_latent_trajectory(encoder, waveform, times):
    """
    Computes dense latent trajectory z_t at the same length as times by interpolating
    the continuous CDE integration states.
    """
    # 1. Waveform Patching
    patched_features = encoder.patcher(waveform) # [batch, T', patcher_out_dim]
    batch_size, T_prime, _ = patched_features.shape
    
    new_times = torch.linspace(times[0].item(), times[-1].item(), T_prime, device=waveform.device)
    time_channel = new_times.unsqueeze(0).unsqueeze(-1).expand(batch_size, T_prime, 1)
    cde_input = torch.cat([time_channel, patched_features], dim=-1)
    
    coeffs = torchcde.natural_cubic_coeffs(cde_input, new_times)
    X = torchcde.CubicSpline(coeffs, new_times)
    
    z0 = encoder.initial_mapping(X.evaluate(new_times[0]))
    z_t = torchcde.cdeint(X=X, z0=z0, func=encoder.cde_func, t=new_times, adjoint=False) # [batch, T_prime, latent_dim]
    
    # Interpolate from T_prime to T (e.g. 1000)
    T_target = times.shape[0]
    z_t_transposed = z_t.transpose(1, 2) # [batch, latent_dim, T_prime]
    z_t_interpolated = torch.nn.functional.interpolate(
        z_t_transposed, size=T_target, mode='linear', align_corners=True
    )
    z_t_final = z_t_interpolated.transpose(1, 2) # [batch, T_target, latent_dim]
    return z_t_final
