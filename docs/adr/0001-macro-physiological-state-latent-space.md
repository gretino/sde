# Macro Physiological State Latent Space

To simulate continuous baseline dynamics ($f_{\text{base}}$) and future ECG rollouts without encountering extreme solver stiffness from high-frequency intra-beat voltage oscillations, we decouple the continuous-time latent space from raw waveform voltages. The latent space canonically represents the continuous macro-level **Physiological State** ($z_t \in \mathbb{R}^{32}$), evolved probabilistically via a conditional Itô Latent Stochastic Differential Equation (SDE) with diagonal diffusion:

$$dz_t = f(t, z_t)\,dt + g(z_t)\,dW_t$$

A dedicated emission decoder maps discrete 25 Hz continuous-time latent trajectory states into reconstructed 100 Hz 12-lead ECG waveforms at query timestamps.
