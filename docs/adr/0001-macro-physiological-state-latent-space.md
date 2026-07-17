# Macro Physiological State Latent Space

To simulate continuous baseline dynamics ($f_{base}$) without encountering extreme ODE solver stiffness from fast intra-beat voltage changes, we decided to decouple the continuous-time latent space from the direct raw waveform path. The latent space canonically represents the macro-level **Physiological State** (capturing slow-moving drivers like autonomic tone and baseline rate drift), while a dedicated waveform decoder maps this instantaneous state to reconstruct high-frequency intra-beat ECG waveforms at query timestamps.
