# Agent To-Do: Verify Whether the ECG-FM + SDE Pipeline Is Working

## Goal

Verify which part of the pipeline is failing or working:

1. ECG-FM latent representation
2. Decoder reconstruction
3. SDE latent dynamics
4. Waveform generation quality
5. Physiological validity of generated ECG

---

## 1. Verify Dataset and Preprocessing

- To test lead correctness, inspect several INCART samples and confirm they contain 12 leads in the expected order used by ECG-FM.
- To test sampling correctness, confirm every ECG segment is resampled to ECG-FM's required sampling rate.
- To test segment shape correctness, confirm each input segment has shape:

  ```text
  [12 leads, segment_length]
  ```

- To test normalization correctness, compute mean and standard deviation per lead after preprocessing and confirm they match the normalization expected by ECG-FM.
- To test target alignment, plot `x_t` and `x_{t+Δt}` for several samples and confirm the future target segment is actually after the input segment by the intended time gap.

---

## 2. Test ECG-FM Encoder Output

- To test latent stability, encode the same ECG segment twice and confirm the latent vectors are identical when the model is in evaluation mode.
- To test latent shape correctness, print the ECG-FM output tensor shape and confirm it matches the expected input shape of the SDE model.
- To test latent scale, compute the mean, standard deviation, minimum, and maximum of ECG-FM latents across the train, validation, and test sets.
- To test train/test consistency, compare latent statistics between train, validation, and test sets and confirm they are similar.

---

## 3. Test Decoder-Only Reconstruction

- To test whether the decoder can reconstruct ECG without the SDE, run:

  ```text
  z_t = ECG-FM(x_t)
  x_hat_t = Decoder(z_t)
  ```

- To evaluate decoder-only reconstruction, compute:

  ```text
  Reconstruction MSE
  Reconstruction MAE
  Pearson correlation
  R-peak F1
  Heart Rate MAE
  HRV RMSSD MAE
  ```

- To verify visual quality, plot at least 20 examples of:

  ```text
  original x_t
  reconstructed x_hat_t
  ```

- To verify QRS preservation, overlay detected R-peaks on both original and reconstructed ECGs.
- To determine whether the decoder is usable, confirm decoder-only reconstruction beats zero-signal, mean-waveform, and random-sample baselines.
- To determine whether waveform generation is possible from ECG-FM latents, confirm decoder-only reconstruction has non-zero R-peak F1 and positive Pearson correlation.

---

## 4. Test Oracle Future Latent Reconstruction

- To test whether perfect future latent input can reconstruct the target future waveform, run:

  ```text
  z_future = ECG-FM(x_{t+Δt})
  x_hat_future_oracle = Decoder(z_future)
  ```

- To evaluate oracle reconstruction, compute:

  ```text
  Reconstruction MSE
  Reconstruction MAE
  Pearson correlation
  R-peak F1
  Heart Rate MAE
  HRV RMSSD MAE
  ```

- To verify visual quality, plot at least 20 examples of:

  ```text
  target future x_{t+Δt}
  oracle reconstruction x_hat_future_oracle
  ```

- To determine whether the decoder bottleneck is the failure point, compare oracle reconstruction against decoder-only reconstruction.
- To determine whether SDE prediction can ever produce valid waveforms, confirm oracle reconstruction produces recognizable ECG morphology and detectable R-peaks.

---

## 5. Build Waveform Baselines

- To test whether the model beats a trivial zero predictor, evaluate:

  ```text
  x_hat = zeros_like(x_target)
  ```

- To test whether the model beats the dataset average, evaluate:

  ```text
  x_hat = mean waveform from training set
  ```

- To test whether the model beats short-term persistence, evaluate:

  ```text
  x_hat_{t+Δt} = x_t
  ```

- To test whether the model beats random guessing, evaluate:

  ```text
  x_hat = randomly selected future segment from another sample
  ```

- For each baseline, compute the same metrics:

  ```text
  MSE
  MAE
  Pearson correlation
  R-peak F1
  Heart Rate MAE
  HRV RMSSD MAE
  ```

- To determine whether the generated waveform is meaningful, confirm the SDE-generated waveform beats all trivial waveform baselines.

---

## 6. Build Latent Dynamics Baselines

- To test whether the SDE beats mean latent prediction, evaluate:

  ```text
  z_hat_{t+Δt} = mean training latent
  ```

- To test whether the SDE beats latent persistence, evaluate:

  ```text
  z_hat_{t+Δt} = z_t
  ```

- To test whether the SDE beats a simple deterministic model, train and evaluate a linear latent predictor:

  ```text
  z_hat_{t+Δt} = A z_t + b
  ```

- To test whether the SDE beats a small neural baseline, train and evaluate an MLP predictor:

  ```text
  z_hat_{t+Δt} = MLP(z_t, Δt)
  ```

- To test whether the SDE beats random latent prediction, evaluate:

  ```text
  z_hat_{t+Δt} = z_random_future
  ```

- For each latent baseline, compute:

  ```text
  Latent MSE
  Latent MAE
  Latent cosine similarity
  Latent R² against persistence baseline
  ```

- To determine whether the SDE learns useful latent dynamics, confirm it beats mean latent, random latent, and persistence latent baselines.

---

## 7. Evaluate SDE Latent Prediction

- To test the SDE latent prediction directly, run:

  ```text
  z_t = ECG-FM(x_t)
  z_target = ECG-FM(x_{t+Δt})
  z_hat = SDE(z_t, Δt)
  ```

- To evaluate latent prediction, compute:

  ```text
  Latent MSE
  Latent MAE
  Cosine similarity between z_hat and z_target
  R² compared with persistence baseline
  ```

- To test prediction quality across time gaps, evaluate separate results for each Δt value.
- To determine whether SDE is useful, confirm latent error increases slower than the persistence baseline as Δt becomes larger.
