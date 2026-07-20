# Variational Latent SDE ELBO and Composite Emission Supervision

To handle the stochasticity of future ECG trajectories given past context while maintaining waveform morphology fidelity, we replace deterministic latent matching with a Variational SDE formulation (ELBO) and composite emission decoding penalties.

During training, a 1D Bidirectional ResNet posterior encoder processes the full (context + future) waveform to produce a posterior initial distribution $q(z_0 \mid x_{\text{context}}, x_{\text{future}})$ and a continuous recognition feature path. The posterior SDE drift $f(t, z_t, c, \text{rec}_t)$ drives continuous trajectory integration.

The system is supervised via:
1. **Laplace Observation Negative Log-Likelihood**: $-\log p(x_{\text{future}} \mid z_{1:T})$ under a learned per-lead log-scale.
2. **Initial-State KL Divergence**: $D_{\mathrm{KL}}[q(z_0) \parallel p(z_0)]$.
3. **Continuous Girsanov Path KL Divergence**: $D_{\mathrm{KL}}[q(z_{1:T}) \parallel p(z_{1:T})]$ penalizing drift divergence between posterior drift $f$ and prior drift $h$.
4. **Composite Waveform Morphology Losses**: Deterministic L1 derivative loss ($\|\Delta \hat{x} - \Delta x\|_1$) and multi-resolution STFT magnitude spectral losses (FFT sizes 32, 64, 128) on the posterior decoded waveform mean.
