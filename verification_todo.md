# SDE Pipeline Verification Checklist & Commands

Use this file to easily resume running and interpreting the pipeline verification results.

## Commands to Execute

To run the verification pipeline on the trained checkpoint (`checkpoints/neurosde_final.pt`) with 200 test samples:

```bash
conda run -n sde python verify_pipeline.py --max-samples 200
```

To run a quick test with fewer samples (e.g., 20 samples) to verify speed and plot output:

```bash
conda run -n sde python verify_pipeline.py --max-samples 20
```

---

## Active TODO Checklist

### 1. Execute & Collect Metrics
- [ ] Run the `verify_pipeline.py` script to completion using the commands above.
- [ ] Verify that `verification_results/verification_metrics_final.json` is generated successfully.
- [ ] Inspect the generated plots under `verification_results/plots/` (original vs reconstructed Lead II signals with R-peak markers).

### 2. Analyze Component Bottlenecks
- [ ] Compare the SDE solver metrics in Section 7 against:
  - **Latent dynamics baselines:** MLP Predictor, Linear Predictor, Persistence, and Mean latent representations.
  - Check if the SDE solver outperforms the deterministic MLP baseline (which got `MSE: 0.08767` in early tests).
- [ ] Check if the trained Pointwise Decoder (`PhaseTolerantDecoder`) in Section 3 and 4 successfully reconstructs ECG signals from latents.
  - *Note:* In the untrained checkpoint, the decoder yielded near-zero correlation ($r \approx -0.002$), which should significantly improve after SDE training.

### 3. Generate Final Report
- [ ] Create a final `verification_report.md` artifact summarizing:
  - Dataset & normalization checks.
  - Waveform reconstruction quality (MSE, MAE, Pearson, R-peak F1, HR/HRV MAE).
  - Latent dynamics comparison table across time gaps $\Delta t \in \{1.0, 2.0, \dots, 10.0\}$.
  - Visual rendering of reconstructed ECG waveforms.
