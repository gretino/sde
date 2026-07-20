# ECG Natural Dynamics Reimplementation Plan

## 1. Objective

Reimplement the repository as a clean conditional latent SDE system for natural ECG learning and future ECG prediction.

The initial milestone will learn the natural temporal evolution of ECG from past waveform context. The milestone includes waveform reconstruction, probabilistic latent dynamics, and short-horizon future prediction. Intervention controls, treatment simulation, music effects, reinforcement learning, and explicit patient-specific adaptation are outside this milestone.

The implementation will replace the existing CDE-to-ODE-to-SIREN pipeline. The repository will contain one coherent model, one training pipeline, one evaluation pipeline, and a focused set of tests.

## 2. Milestone Definition

The completed system will:

1. Read a past 12-lead ECG context window.
2. Infer a probabilistic latent state at the forecasting boundary.
3. Model the future latent trajectory with a conditional latent SDE.
4. Decode the latent trajectory into a future 12-lead ECG waveform.
5. Reconstruct observed future ECG through a posterior latent process during training.
6. Forecast future ECG through a context-only prior latent process during validation and inference.
7. Produce multiple coherent future samples from the same context.
8. Preserve ECG rhythm, R-peaks, and waveform morphology over a two-second forecast horizon.

The first full milestone will use:

- INCART ECG records
- 100 Hz sampling rate
- 5-second context windows
- 2-second future windows
- 12 leads
- 1-second window stride
- 25 Hz latent trajectory rate
- 32-dimensional latent state

A Lead II-only debug configuration will be included for overfitting and gradient verification. The production milestone remains 12-lead forecasting.

## 3. Repository Replacement

Remove the current modeling and experiment implementation:

```text
sde/baseline.py
sde/encoder.py
sde/solver.py
sde/loss.py
sde/patching.py
sde/weight_utils.py
run_incart_experiment.py
pretrain_autoencoder.py
overfit_single.py
verify_pipeline.py
verify_single.py
config/pretrain_config.yaml
config/stage1_config.yaml
config/stage2_config.yaml
```

Remove the current ODE-specific tests and architecture decision records.

Preserve the useful ECG preprocessing logic and rewrite it under the new package structure. Preserve the INCART loading behavior only where it remains compatible with the new data contract.

Create the following repository layout:

```text
/
├── pyproject.toml
├── README.md
├── configs/
│   ├── debug_lead2.yaml
│   └── incart_12lead.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── sample_forecasts.py
│   └── inspect_batch.py
├── src/
│   └── ecg_forecast/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── incart.py
│       │   ├── preprocessing.py
│       │   ├── windows.py
│       │   └── collate.py
│       ├── models/
│       │   ├── context_encoder.py
│       │   ├── posterior_encoder.py
│       │   ├── conditional_sde.py
│       │   ├── emission_decoder.py
│       │   └── latent_sde_forecaster.py
│       ├── losses/
│       │   ├── elbo.py
│       │   ├── morphology.py
│       │   └── schedules.py
│       ├── metrics/
│       │   ├── waveform.py
│       │   ├── rhythm.py
│       │   └── uncertainty.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── checkpointing.py
│       │   └── logging.py
│       └── visualization/
│           └── forecasts.py
└── tests/
    ├── test_dataset.py
    ├── test_model_shapes.py
    ├── test_sde_gradients.py
    ├── test_posterior_overfit.py
    ├── test_prior_forecast.py
    ├── test_sampling.py
    └── test_no_future_leakage.py
```

Use an installable `src` layout. All scripts will import from `ecg_forecast`.

## 4. Dependencies

Define the project dependencies in `pyproject.toml`:

```text
torch
torchsde
wfdb
numpy
scipy
pyyaml
tqdm
wandb
neurokit2
matplotlib
pytest
```

Use `torchsde` as the only differential-equation solver dependency.

## 5. Data Pipeline

### 5.1 Record Split

Split records before window construction:

```text
50 training records
10 validation records
15 test records
```

Use seed `42` and save the exact split lists to:

```text
artifacts/splits/incart_seed42.json
```

Every window from one record must remain in the same split.

### 5.2 Window Construction

Construct overlapping windows with:

```text
context duration: 5.0 seconds
future duration: 2.0 seconds
sampling rate: 100 Hz
context samples: 500
future samples: 200
stride: 1.0 second
```

Each dataset item will return:

```python
{
    "record_id": str,
    "context_waveform": FloatTensor[500, num_leads],
    "future_waveform": FloatTensor[200, num_leads],
    "context_times": FloatTensor[500],
    "future_times": FloatTensor[200],
    "normalization_mean": FloatTensor[num_leads],
    "normalization_std": FloatTensor[num_leads],
    "future_r_peaks": LongTensor[num_peaks],
}
```

### 5.3 Normalization

Compute normalization statistics from the context window only:

```python
mean = context_waveform.mean(dim=0)
std = context_waveform.std(dim=0).clamp_min(1e-5)
```

Apply the same context-derived mean and standard deviation to both context and future waveforms.

Store the statistics in each sample for inverse transformation during plotting and metric calculation.

### 5.4 Preprocessing

Resample every record to 100 Hz.

Preserve all 12 leads in a fixed channel order.

Use the original annotations to produce future-window R-peak indices after resampling.

Cache preprocessed records by:

```text
record ID
target sampling rate
preprocessing version
```

Include the preprocessing version in the cache filename so stale cache files cannot silently survive code changes.

## 6. Model Architecture

### 6.1 Context Encoder

Implement a causal 1D residual CNN.

Input:

```text
[B, 500, 12]
```

Internal channel-first representation:

```text
[B, 12, 500]
```

Architecture:

```text
Conv1d(12, 64, kernel_size=7, stride=2, padding=3)
GELU
ResidualBlock(64, 64)
Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
GELU
ResidualBlock(128, 128)
ResidualBlock(128, 128)
```

The output temporal rate is 25 Hz:

```text
[B, 125, 128]
```

Apply attention pooling over the 125 context tokens to produce:

```text
context_summary: [B, 128]
```

Project the context summary into the conditional prior initial distribution:

```text
prior_mean: [B, 32]
prior_logvar: [B, 32]
```

Clamp `prior_logvar` to:

```text
[-8, 4]
```

### 6.2 Posterior Encoder

Implement a bidirectional 1D residual CNN over the concatenated context and future waveform.

Input:

```text
[B, 700, 12]
```

Use the same downsampling factor of four and output width of 128.

Produce:

```text
posterior_summary: [B, 128]
recognition_path: [B, 50, 128]
posterior_mean: [B, 32]
posterior_logvar: [B, 32]
```

The recognition path corresponds only to the 2-second future interval at 25 Hz.

Project the combined context and posterior summaries into the posterior initial distribution.

The posterior encoder will be called only during training and posterior-reconstruction evaluation.

### 6.3 Conditional Latent SDE

Implement an Itô SDE with diagonal noise:

\[
dz_t = f(t, z_t)\,dt + g(z_t)\,dW_t.
\]

The module will expose:

```python
f(t, z)  # posterior drift
h(t, z)  # prior drift
g(t, z)  # shared diagonal diffusion
```

#### Prior Drift

Condition the prior drift on:

```text
current latent state
continuous time
context summary
```

Use:

```text
Linear(32 + 1 + 128, 128)
Tanh
Linear(128, 128)
Tanh
Linear(128, 32)
```

#### Posterior Drift

Condition the posterior drift on:

```text
current latent state
continuous time
context summary
interpolated recognition feature
```

Use:

```text
Linear(32 + 1 + 128 + 128, 128)
Tanh
Linear(128, 128)
Tanh
Linear(128, 32)
```

Interpolate the 50-step recognition path at the solver’s current continuous time.

#### Diffusion

Use one learned diffusion value per latent dimension:

```python
sigma = 1e-4 + softplus(raw_sigma)
```

Initialize the effective diffusion standard deviation to `0.03`.

The diffusion remains state-independent for the initial milestone.

#### Integration

Use:

```text
torchsde.sdeint
method: euler
dt: 0.01 seconds
noise type: diagonal
SDE type: Ito
```

Evaluate the latent process at 50 future timestamps:

```text
0.04, 0.08, ..., 2.00 seconds
```

Return:

```text
future_latent_path: [B, 50, 32]
path_kl: scalar
```

Implement the path KL with the Girsanov drift-difference term using the shared diffusion.

### 6.4 Emission Decoder

Decode each 25 Hz latent state into four 100 Hz ECG samples.

Condition the decoder on:

```text
latent state
context summary
```

Architecture:

```text
Linear(32 + 128, 256)
GELU
Linear(256, 256)
GELU
Linear(256, 4 * num_leads)
```

Reshape the output:

```text
[B, 50, 4, num_leads]
```

Then flatten it into:

```text
[B, 200, num_leads]
```

The decoder will output the waveform mean.

Maintain one learned observation log-scale per lead:

```text
observation_log_scale: [num_leads]
```

Clamp the resulting scale to:

```text
[0.01, 2.0]
```

Use a Laplace observation distribution.

The decoder receives no explicit normalized time coordinate.

### 6.5 Complete Model Interface

Implement:

```python
class LatentSDEForecaster(nn.Module):
    def forward_posterior(
        self,
        context_waveform,
        future_waveform,
        context_times,
        future_times,
        brownian_motion=None,
    ) -> ForecastOutput:
        ...

    def forward_prior(
        self,
        context_waveform,
        context_times,
        future_times,
        num_samples=1,
        brownian_motion=None,
    ) -> ForecastOutput:
        ...
```

Define:

```python
@dataclass
class ForecastOutput:
    waveform_mean: torch.Tensor
    waveform_scale: torch.Tensor
    latent_path: torch.Tensor
    initial_kl: torch.Tensor | None
    path_kl: torch.Tensor | None
```

`forward_posterior` will reconstruct the observed future and return KL terms.

`forward_prior` will forecast without receiving the future waveform.

## 7. Loss Function

### 7.1 Observation Likelihood

Use the mean negative log-likelihood of the target under the Laplace observation distribution:

\[
L_{\text{nll}}
=
-\mathbb{E}_{q}
\left[
\log p(x_{\text{future}}\mid z_{1:T})
\right].
\]

Average over batch, time, and leads.

### 7.2 Initial-State KL

Compute:

\[
L_{\text{initial-kl}}
=
D_{\mathrm{KL}}
\left[
q(z_0\mid x_{\text{context}},x_{\text{future}})
\parallel
p(z_0\mid x_{\text{context}})
\right].
\]

Average over batch and latent dimensions.

### 7.3 Path KL

Compute the normalized path KL from the posterior and prior drift difference.

Average over batch, latent dimensions, and future time.

### 7.4 Morphology Loss

Add two deterministic auxiliary losses on the posterior waveform mean.

#### Derivative Loss

\[
L_{\text{derivative}}
=
\left\|
\Delta \hat{x}
-
\Delta x
\right\|_1.
\]

#### Multi-Resolution Spectral Loss

Calculate magnitude STFT losses with:

```text
FFT sizes: 32, 64, 128
hop sizes: 8, 16, 32
```

Average the three L1 magnitude losses.

### 7.5 Total Loss

Use:

\[
L =
L_{\text{nll}}
+
\beta_{\text{initial}}L_{\text{initial-kl}}
+
\beta_{\text{path}}L_{\text{path-kl}}
+
0.5L_{\text{derivative}}
+
0.1L_{\text{spectral}}.
\]

Normalize every loss term before weighting.

## 8. Training Schedule

Use three explicit stages within one training script.

### 8.1 Stage A: Posterior Reconstruction Warmup

Duration:

```text
10 epochs
```

Train:

```text
context encoder
posterior encoder
posterior drift
diffusion
emission decoder
observation scale
```

Use:

```text
beta_initial = 0
beta_path = 0
```

This stage establishes that the latent path and decoder can represent natural ECG morphology.

Save:

```text
checkpoints/posterior_warmup_best.pt
```

Selection metric:

```text
validation posterior waveform NLL
```

### 8.2 Stage B: Prior Alignment

Duration:

```text
20 epochs
```

Train all model parameters.

Linearly increase:

```text
beta_initial: 0.0 → 1.0 over 10 epochs
beta_path: 0.0 → 1.0 over 10 epochs
```

Keep both values at `1.0` for the remaining ten epochs.

Save:

```text
checkpoints/prior_alignment_best.pt
```

Selection metric:

```text
validation prior forecast score
```

Define the forecast score as:

\[
\text{score}
=
\text{prior NLL}
+
(1-\text{Pearson})
+
(1-\text{R-peak F1}).
\]

### 8.3 Stage C: Forecast Refinement

Duration:

```text
20 epochs
```

Train all model parameters with the full loss and fixed KL weights.

Reduce the learning rate by a factor of ten.

Save:

```text
checkpoints/final_best.pt
```

### 8.4 Optimization

Use:

```text
optimizer: AdamW
initial learning rate: 3e-4
weight decay: 1e-4
gradient clipping: global norm 1.0
batch size: 32
mixed precision: enabled
seed: 42
```

Use cosine learning-rate decay within each stage.

Track the best checkpoint separately for every stage.

## 9. Validation and Evaluation

Report posterior reconstruction and prior forecasting as separate tasks.

### 9.1 Posterior Reconstruction Metrics

Calculate:

```text
Laplace negative log-likelihood
MSE
MAE
Pearson correlation
R-peak F1
heart-rate MAE
RMSSD MAE
derivative MAE
spectral magnitude MAE
```

### 9.2 Prior Forecast Metrics

Draw 16 prior trajectories for each context.

Report:

```text
mean-sample NLL
ensemble-mean MSE
ensemble-mean MAE
ensemble-mean Pearson correlation
median sample R-peak F1
median sample heart-rate MAE
median sample RMSSD MAE
90% interval coverage
90% interval width
```

### 9.3 Baseline

Implement a repeat-context baseline.

Construct the forecast by repeating the final two seconds of the context window.

Evaluate the baseline with the same deterministic waveform and rhythm metrics.

The latent SDE milestone must outperform this baseline on:

```text
Pearson correlation
R-peak F1
heart-rate MAE
```

### 9.4 Visualization

Save one fixed validation example and one fixed test example after every epoch.

Produce four panels for Lead II:

1. Context waveform.
2. Ground-truth future waveform.
3. Posterior reconstruction.
4. Prior forecast samples with ensemble mean and 90% interval.

Also save a 12-lead page for the final checkpoint.

## 10. Tests

### 10.1 Dataset Test

Verify:

```text
correct tensor shapes
record-level split isolation
context-only normalization
correct future R-peak indexing
deterministic split generation
```

### 10.2 Shape Test

Verify model outputs for:

```text
Lead II debug mode
12-lead mode
posterior forward pass
prior forward pass
multiple prior samples
```

### 10.3 Gradient Test

Run one training step and verify finite, nonzero gradients for:

```text
context encoder
posterior encoder
prior drift
posterior drift
diffusion
decoder
prior initial distribution head
posterior initial distribution head
observation scale
```

### 10.4 Posterior Overfit Test

Overfit one batch in Lead II mode.

Acceptance criteria:

```text
posterior Pearson correlation >= 0.95
posterior R-peak F1 >= 0.95
finite initial KL
finite path KL
```

### 10.5 Prior Forecast Test

Train on one fixed batch through all three stages.

Verify:

```text
prior forward pass receives no future waveform
prior output is nonconstant
prior latent temporal standard deviation is nonzero
prior forecast improves over its initialization
```

### 10.6 Sampling Test

Use a fixed context and different Brownian paths.

Verify:

```text
different Brownian paths produce different latent trajectories
samples preserve similar overall rhythm
fixed Brownian path and fixed seed reproduce the same result
```

### 10.7 Leakage Test

Instrument the model and dataset so the prior path has no access to:

```text
future waveform
future-derived normalization statistics
posterior recognition features
future R-peak annotations
```

## 11. Debugging Gates

Development will proceed through the following gates.

### Gate 1: Data

Pass all dataset tests and inspect saved context/future plots.

### Gate 2: Posterior Mechanics

Overfit one Lead II batch to the posterior acceptance criteria.

### Gate 3: Prior Mechanics

Produce nonconstant prior forecasts with finite KL and nonzero gradients.

### Gate 4: Small Dataset

Train on five records and validate on two records.

Confirm that posterior reconstruction remains strong and prior forecasting improves over the repeat-context baseline.

### Gate 5: Full 12-Lead Training

Train on the complete 50/10/15 record split.

A later gate begins only after the previous gate passes.

## 12. Configuration

Create `configs/debug_lead2.yaml`:

```yaml
seed: 42
sampling_rate: 100
context_seconds: 5.0
future_seconds: 2.0
stride_seconds: 1.0
lead_indices: [1]
latent_rate: 25
latent_dim: 16
context_dim: 64
batch_size: 8
learning_rate: 0.0003
weight_decay: 0.0001
posterior_warmup_epochs: 200
prior_alignment_epochs: 200
forecast_refinement_epochs: 100
num_workers: 0
mixed_precision: false
```

Create `configs/incart_12lead.yaml`:

```yaml
seed: 42
sampling_rate: 100
context_seconds: 5.0
future_seconds: 2.0
stride_seconds: 1.0
lead_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
latent_rate: 25
latent_dim: 32
context_dim: 128
batch_size: 32
learning_rate: 0.0003
weight_decay: 0.0001
posterior_warmup_epochs: 10
prior_alignment_epochs: 20
forecast_refinement_epochs: 20
num_workers: 4
mixed_precision: true
prior_samples_eval: 16
```

## 13. Checkpoint Format

Save:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "stage": ...,
    "epoch": ...,
    "global_step": ...,
    "config": ...,
    "validation_metrics": ...,
    "record_splits": ...,
    "preprocessing_version": ...,
}
```

Loading a checkpoint must restore the complete model and stage state.

## 14. Logging

Use one W&B project:

```text
ecg-natural-dynamics
```

Log:

```text
total loss
observation NLL
initial KL
path KL
derivative loss
spectral loss
posterior Pearson
prior Pearson
posterior R-peak F1
prior R-peak F1
diffusion mean
diffusion minimum
diffusion maximum
latent temporal standard deviation
gradient norm by module
forecast visualizations
```

## 15. Completion Criteria

The initial milestone is complete when all of the following conditions are met:

1. The repository contains only the new implementation and its supporting data, training, evaluation, and test code.
2. All automated tests pass.
3. The Lead II posterior overfit test reaches Pearson correlation and R-peak F1 of at least `0.95`.
4. The 12-lead posterior reconstruction reaches validation Pearson correlation of at least `0.85`.
5. The prior forecast produces recognizable ECG waveforms instead of constant or smooth-line collapse.
6. The prior forecast outperforms the repeat-context baseline on validation Pearson correlation, R-peak F1, and heart-rate MAE.
7. Different Brownian paths produce distinct but rhythmically coherent future trajectories.
8. The prior forecasting path uses only the context waveform and future timestamps.
9. The final checkpoint reproduces its reported test metrics from the evaluation script.
10. The README contains exact installation, training, evaluation, and forecast-sampling commands.

## 16. Scope Boundary

This milestone contains natural ECG learning and natural future prediction.

The milestone excludes:

```text
music controls
medication controls
treatment effects
counterfactual claims
reinforcement learning
causal intervention estimation
explicit patient embeddings
online patient adaptation
long-horizon disease progression
```

These capabilities will be designed after the natural forecasting milestone passes its acceptance criteria.
