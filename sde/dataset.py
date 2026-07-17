import torch

class SegmentBuilder:
    def __init__(self, sampling_rate: int, context_window: float, prediction_window: float):
        """
        Initializes the SegmentBuilder which maps continuous ECG records into
        discrete context and target segments, enforcing a Normalized Anchor Time.
        
        Args:
            sampling_rate: Hz
            context_window: Length of the context window in seconds
            prediction_window: Length of the prediction window in seconds
        """
        self.sampling_rate = sampling_rate
        self.context_window = context_window
        self.prediction_window = prediction_window
        
        self.dt = 1.0 / self.sampling_rate
        self.context_pts = int(self.context_window * self.sampling_rate)
        self.target_pts = int(self.prediction_window * self.sampling_rate)

    def build_segment(self, raw_waveform: torch.Tensor, start_time_sec: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts context and target windows from the raw waveform starting at start_time_sec.
        
        Returns:
            context_wf: [context_pts, leads]
            context_t: [context_pts] with context_t[-1] == 0.0
            target_wf: [target_pts, leads]
            target_t: [target_pts] with target_t > 0.0
        """
        start_idx = int(start_time_sec * self.sampling_rate)
        
        # Slicing
        context_end_idx = start_idx + self.context_pts
        target_end_idx = context_end_idx + self.target_pts
        
        context_wf = raw_waveform[start_idx:context_end_idx]
        target_wf = raw_waveform[context_end_idx:target_end_idx]
        
        # Time mapping (ADR-0002)
        # Context ends at exactly t=0
        context_start_t = -(self.context_pts - 1) * self.dt
        context_t = torch.linspace(context_start_t, 0.0, self.context_pts)
        
        # Target begins immediately after t=0
        target_end_t = self.target_pts * self.dt
        target_t = torch.linspace(self.dt, target_end_t, self.target_pts)
        
        return context_wf, context_t, target_wf, target_t
