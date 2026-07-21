# 12-Lead ECG Latent Dynamics Pipeline Improvement and Debugging Plan

## 1. Objective

Revise the 12-lead ECG latent dynamics prototype so that failures can be isolated before another full stochastic SDE training run.

The current system has demonstrated that the future-conditioned posterior can reconstruct 12-lead ECG. It has not yet demonstrated that the context-only prior can autonomously predict a physiologically meaningful future trajectory.

The next implementation will separate three questions:

1. Can the current data and loss support direct 12-lead future prediction?
2. Can a deterministic latent prior predict a useful mean future trajectory?
3. Can stochasticity be added without collapsing the mean forecast or the predictive distribution?

The pipeline will proceed through explicit diagnostic and acceptance gates. Do not continue to stochastic SDE training until the deterministic prior produces recognizable ECG.

---

## 2. Current Failure Hypotheses

The current prior forecast may be failing for one or more of the following reasons:

```text
context representation loses cardiac boundary phase
prior initial state is incorrect
prior drift fails during autonomous rollout
decoder ignores the latent trajectory
decoder relies too heavily on the global context summary
four-sample decoder blocks create poor temporal continuity
teacher and student paths are both stochastic during alignment
the Stage B teacher is not completely frozen
pointwise stochastic waveform loss encourages mean and variance collapse
posterior latent targets contain future information unavailable from context
two-second forecasting is too difficult as the first deterministic target
```

The debugging scripts must distinguish these failure modes rather than treating all flat forecasts as one generic collapse problem.

---

# Part I: Main Pipeline Changes

## 3. Preserve the Current Posterior as a Diagnostic Reference

Keep the existing Stage A posterior checkpoint for diagnostics.

The posterior will be used as:

```text
a reconstruction ceiling
a latent teacher
a source of oracle initial states
a source of oracle future-conditioned trajectories
```

Do not treat posterior reconstruction performance as proof of forecasting ability.

Retrain Stage A only after changing:

```text
context encoder architecture
posterior encoder architecture
latent dimensionality
decoder architecture
latent temporal rate
normalization behavior
```

## 4. Replace Stage B with Deterministic Prior Training

The immediate Stage B objective is to learn one valid context-only mean trajectory.

Disable stochasticity during this phase.

Use:

```text
prior initial state = prior mean
posterior initial state = posterior mean
diffusion = 0
initial-state sampling = disabled
observation sampling = disabled
```

Freeze:

```text
entire posterior encoder
entire posterior drift
entire decoder
entire context feature extractor
entire context pooling module
posterior initial-state heads
diffusion parameters
observation-scale parameters
prior log-variance head
```

Train only:

```text
prior initial-state mean head
prior drift
```

Create a separate frozen teacher copy or ensure that every teacher parameter and teacher input representation is unchanged during Stage B.

The Stage B teacher must not move between optimizer steps.

## 5. Use Deterministic Teacher Targets

Use detached posterior means instead of sampled posterior states.

Define:

\[
z_{0}^{q} = \mu_q
\]

and:

\[
z_{0}^{p} = \mu_p.
\]

Run the posterior teacher with:

```text
posterior mean initial state
zero diffusion
future recognition path enabled
```

Run the prior student with:

```text
prior mean initial state
zero diffusion
context-only prior drift
```

This removes variance-related noise while testing whether the prior can learn the mean dynamics.

## 6. Add Autonomous Trajectory Matching

The current drift matching loss evaluates the prior drift on posterior teacher states. This is teacher forcing and does not ensure stable autonomous prior rollout.

Add trajectory matching:

\[
L_{\text{trajectory}}
=
\frac{1}{BTD}
\left\|
z_{1:T}^{p}
-
\operatorname{stopgrad}
\left(
z_{1:T}^{q}
\right)
\right\|_2^2.
\]

The prior path \(z^p\) must be generated from its own initial state and its own previous states.

Keep drift imitation as a secondary loss:

\[
L_{\text{drift}}
=
\frac{1}{BTD}
\left\|
h(t,z_t^q)
-
\operatorname{stopgrad}
\left(
f(t,z_t^q)
\right)
\right\|_2^2.
\]

Use direct initial-state mean matching:

\[
L_{z_0}
=
\frac{1}{BD}
\left\|
\mu_p
-
\operatorname{stopgrad}(\mu_q)
\right\|_2^2.
\]

Do not match posterior and prior variance during deterministic Stage B.

## 7. Revised Deterministic Stage B Loss

Use:

\[
L_B
=
L_{\text{prior-waveform}}
+
\lambda_{\text{trajectory}}L_{\text{trajectory}}
+
\lambda_{z_0}L_{z_0}
+
\lambda_{\text{drift}}L_{\text{drift}}.
\]

Initial weights:

```text
prior waveform weight: 1.0
trajectory weight: 1.0
initial-state mean weight: 0.1
drift weight: 0.01
```

The weights must be logged separately.

Do not use:

```text
initial-state KL
path KL
prior variance matching
learnable diffusion
multisample waveform loss
```

during deterministic Stage B.

## 8. Add Forecast-Horizon Curriculum

Do not begin with the full two-second forecast.

Train and evaluate sequentially:

```text
Stage B1: 0.5-second future, 50 waveform samples
Stage B2: 1.0-second future, 100 waveform samples
Stage B3: 2.0-second future, 200 waveform samples
```

A longer horizon begins only after the shorter horizon passes its acceptance gate.

The context remains five seconds and all 12 leads remain active.

## 9. Improve the Context Representation

The existing global attention summary may not preserve cardiac phase at the context boundary.

The context encoder must output:

```text
global_summary
boundary_token
recent_summary
context_tokens
```

Define:

```text
global_summary:
    attention pooling over the complete five-second context

boundary_token:
    final context token

recent_summary:
    GRU, temporal attention, or average pooling over the final 1–2 seconds
```

Construct the prior dynamic context:

\[
c_{\text{dynamic}}
=
[
c_{\text{global}},
c_{\text{boundary}},
c_{\text{recent}}
].
\]

Use `dynamic_context` for:

```text
prior initial-state mean
prior drift conditioning
future R-peak timing head
```

Use `global_summary` primarily for static morphology conditioning.

Do not discard the complete token sequence before testing whether recent tokens improve future timing prediction.

## 10. Add Explicit Cardiac Phase Supervision

Add a future R-peak probability head from the latent trajectory.

Pipeline:

```text
latent path at 25 Hz
temporal upsampling to 100 Hz
small Conv1d prediction head
future R-peak probability
```

Create soft R-peak targets by placing a small Gaussian around each annotated future R-peak.

Use:

\[
L_{\text{rhythm}}
=
\operatorname{BCE}
(\hat{r}_{1:T},r_{1:T}).
\]

Add to the deterministic waveform objective:

\[
L_{\text{prior-waveform}}
=
L_{\text{Laplace}}
+
0.5L_{\text{derivative}}
+
0.1L_{\text{spectral}}
+
0.5L_{\text{rhythm}}.
\]

The rhythm loss prevents the model from minimizing waveform loss by producing a smooth baseline without cardiac events.

## 11. Replace the Four-Sample Independent Decoder

The current decoder maps each 25 Hz latent state independently into four 100 Hz waveform samples.

Replace this with a temporal decoder:

```text
latent path [B, 50, D]
linear projection to decoder channels
linear interpolation to [B, 200, C]
three residual temporal Conv1d blocks
final Conv1d to 12 leads
```

Use temporal kernels across adjacent latent steps so waveform continuity is modeled explicitly.

Condition static morphology using FiLM:

```text
global context summary
→ scale and bias parameters
→ temporal decoder residual blocks
```

Do not concatenate the entire global context vector to every latent step as the only conditioning mechanism.

The decoder must be forced to obtain rhythm and temporal progression from the latent path.

## 12. Establish a Direct Forecasting Baseline

Add a deterministic 12-lead TCN baseline:

\[
x_{-5:0}
\rightarrow
\hat{x}_{0:T}.
\]

Use the same:

```text
record split
window construction
normalization
forecast horizons
waveform losses
rhythm loss
evaluation metrics
```

This baseline determines whether the context-to-future task is learnable without latent dynamics.

Interpretation:

```text
TCN succeeds, latent prior fails:
    latent pipeline or training objective is the problem

TCN and latent prior both fail:
    data alignment, context representation, target horizon, or loss is the problem

TCN succeeds at 0.5 seconds but fails at 2 seconds:
    use horizon curriculum and reconsider the two-second target

TCN cannot overfit a tiny set:
    investigate data, target construction, normalization, and loss implementation
```

## 13. Restore Stochastic SDE Training Only After Deterministic Success

Restore stochasticity only when the deterministic prior passes the two-second acceptance gate.

Reintroduce in this order:

```text
1. prior initial-state log-variance
2. initial-state sampling
3. bounded diffusion
4. path KL
5. multisample calibration evaluation
```

Use the posterior likelihood for waveform reconstruction.

Use prior-posterior distribution matching for uncertainty alignment.

Retain a deterministic prior-mean waveform anchor:

\[
L_{\text{prior-mean}}
=
L_{\text{waveform}}
\left(
D(z_{\text{mean}}^p),x_f
\right).
\]

Do not optimize the mean of pointwise waveform losses over many independent stochastic prior samples against the same future target.

That objective directly rewards predictive variance collapse.

---

# Part II: Required Debugging Scripts

## 14. `scripts/diagnose_rollout_components.py`

### Purpose

Identify whether the failure is caused by:

```text
prior initial state
prior autonomous drift
decoder behavior
```

### Required Rollouts

For each validation context, run deterministic zero-diffusion paths.

#### A. Full posterior reconstruction

```text
posterior mean initial state
posterior drift
future recognition path
decoder
```

#### B. Full prior forecast

```text
prior mean initial state
prior drift
decoder
```

#### C. Oracle initial-state rollout

```text
posterior mean initial state
prior drift
decoder
```

#### D. Oracle future-dynamics rollout

```text
prior mean initial state
posterior drift
future recognition path
decoder
```

#### E. Direct teacher latent decoding

```text
posterior latent path
decoder
```

### Required Metrics

Record for every rollout:

```text
12-lead MSE
12-lead MAE
macro lead-wise Pearson
per-lead Pearson
R-peak precision
R-peak recall
R-peak F1
heart-rate MAE
waveform temporal standard deviation
waveform amplitude range
latent temporal standard deviation
```

Record metrics separately for:

```text
0–0.25 seconds
0–0.5 seconds
0–1.0 seconds
0–2.0 seconds
```

### Interpretation

```text
C succeeds and B fails:
    prior initial-state encoder is the main failure

C fails:
    prior autonomous drift is the main failure

D succeeds and B fails:
    prior drift and/or prior initial state are failing

latent trajectory is dynamic but decoded waveform is flat:
    decoder ignores relevant latent dimensions

A and E succeed while B, C, and D fail:
    future-conditioned teacher is valid but context-only prior is not learnable in its current form
```

## 15. `scripts/diagnose_decoder_dependence.py`

### Purpose

Determine whether the decoder uses the latent trajectory or bypasses it through context conditioning.

### Required Decoder Inputs

Decode:

```text
posterior latent + correct context
posterior latent + zero context
zero latent + correct context
time-shuffled posterior latent + correct context
batch-shuffled posterior latent + correct context
posterior latent + batch-shuffled context
prior latent + correct context
```

### Gradient Sensitivity

Calculate:

```text
mean absolute gradient of waveform with respect to latent path
mean absolute gradient of waveform with respect to context summary
gradient ratio: latent sensitivity / context sensitivity
```

Calculate per latent dimension:

```text
decoder output sensitivity to latent dimension d
```

### Required Metrics

For every ablation:

```text
12-lead MSE
12-lead MAE
macro lead-wise Pearson
per-lead Pearson
R-peak F1
waveform temporal standard deviation
waveform amplitude range
```

### Interpretation

```text
zero latent + correct context performs well:
    decoder context shortcut

time-shuffled latent performs similarly to original:
    decoder ignores temporal order

batch-shuffled latent performs similarly:
    decoder ignores latent identity

zero context preserves rhythm but changes morphology:
    desired separation of temporal and static information

zero context destroys all output:
    context conditioning dominates excessively

latent gradient sensitivity is near zero:
    decoder does not use the latent path
```

## 16. `scripts/diagnose_uncertainty_sources.py`

### Purpose

Locate where multisample uncertainty disappears.

### Sampling Conditions

For one fixed context, generate at least 128 trajectories.

#### Initial-state variation only

```text
sample prior z0
diffusion = 0
```

#### Brownian variation only

```text
z0 = prior mean
sample Brownian paths
```

#### Combined variation

```text
sample prior z0
sample Brownian paths
```

#### Fully deterministic reference

```text
z0 = prior mean
diffusion = 0
```

### Required Metrics

#### Initial distribution

```text
prior mean across latent dimensions
prior mean standard deviation
prior log-variance mean
prior log-variance minimum
prior log-variance maximum
fraction of prior log-variance values at lower clamp
posterior log-variance mean/min/max
fraction of posterior log-variance values at lower clamp
empirical z0 standard deviation across samples
```

#### Latent-path uncertainty

```text
mean cross-sample latent standard deviation at z0
mean cross-sample latent standard deviation per future step
final-step latent standard deviation
mean pairwise latent-path distance
latent variance retention ratio
```

Define:

\[
\text{latent variance retention}
=
\frac{
\operatorname{Var}(z_T)
}{
\operatorname{Var}(z_0)+\epsilon
}.
\]

#### Waveform uncertainty

```text
mean cross-sample waveform standard deviation
per-lead waveform standard deviation
mean pointwise 90% interval width
maximum pointwise 90% interval width
mean pairwise waveform distance
fraction of samples with zero detected R-peaks
R-peak count variance
heart-rate variance
```

### Interpretation

```text
z0 variance is near zero:
    prior variance collapse

z0 variance exists but latent variance rapidly vanishes:
    prior drift is strongly contracting

latent variance exists but waveform variance is near zero:
    decoder is insensitive to latent variation

waveform variance exists but samples are not ECG-like:
    uncertainty exists around an invalid mean trajectory

initial-state variation produces diversity but Brownian variation does not:
    diffusion is too small or ignored

Brownian variation produces diversity but initial-state variation does not:
    prior log-variance has collapsed
```

## 17. `scripts/probe_context_phase.py`

### Purpose

Test whether the context representation contains enough timing information to predict the immediate future.

### Frozen Representations

Extract:

```text
global attention summary
final context token
mean of final 25 tokens
mean of final 50 tokens
GRU summary of final 50 tokens
global summary + final token
global summary + recent summary
```

### Probe Targets

Train small linear or two-layer MLP probes to predict:

```text
time from context boundary to next R-peak
last RR interval
median context RR interval
first future RR interval
number of R-peaks in future window
future mean heart rate
```

Use regression for timing and RR targets.

Use classification or Poisson-style count prediction for future peak count.

### Required Metrics

```text
next-R-peak timing MAE in milliseconds
next-R-peak timing median absolute error
next-R-peak timing R²
last-RR MAE
first-future-RR MAE
future heart-rate MAE
future peak-count accuracy
future peak-count macro F1
```

### Interpretation

```text
global summary is weak and final/recent tokens are strong:
    replace global-only prior conditioning

all context features are weak:
    context encoder does not preserve timing information

timing probe is accurate but prior forecast is flat:
    failure is downstream of the context encoder

timing probe is inaccurate even on training data:
    context architecture or preprocessing is insufficient
```

## 18. `scripts/overfit_deterministic_prior.py`

### Purpose

Verify that the deterministic prior can memorize a tiny 12-lead dataset.

### Configuration

Use:

```text
32 training windows
no validation generalization requirement
all 12 leads
0.5-second horizon first
prior mean initial state
zero diffusion
no latent sampling
no KL
fixed decoder and teacher for initial test
batch size 8 or 16
```

Run two versions:

```text
Version A:
    train prior mean head and prior drift only

Version B:
    train context encoder, prior mean head, prior drift, and decoder
```

### Required Metrics

Record per epoch:

```text
training MSE
training MAE
macro Pearson
per-lead Pearson
R-peak F1
heart-rate MAE
trajectory teacher MSE
initial-state mean MSE
latent temporal standard deviation
gradient norms by module
```

### Acceptance Criteria

For 0.5-second overfit:

```text
waveform visually resembles target in all 12 leads
macro Pearson >= 0.90
R-peak F1 >= 0.90
latent temporal standard deviation remains nonzero
no flat-line output
```

Failure indicates an architecture, implementation, gradient, or target-alignment issue.

## 19. `scripts/train_direct_tcn_baseline.py`

### Purpose

Determine whether the raw context-to-future task is learnable without the latent SDE.

### Architecture

Use a deterministic temporal convolutional baseline:

```text
12-lead input
causal residual Conv1d encoder
temporal bottleneck
upsampling decoder
12-lead future output
```

Use the same horizons:

```text
0.5 seconds
1.0 second
2.0 seconds
```

### Required Metrics

Use the same forecast metrics as the latent prior:

```text
MSE
MAE
macro lead-wise Pearson
per-lead Pearson
R-peak precision/recall/F1
heart-rate MAE
waveform temporal standard deviation
waveform amplitude range
```

### Interpretation

```text
TCN succeeds but latent prior fails:
    latent pipeline is responsible

TCN and latent prior both fail:
    task formulation, phase loss, or data alignment is responsible

TCN overfits tiny data but fails validation:
    generalization or dataset diversity issue

TCN cannot overfit tiny data:
    preprocessing, target construction, or loss bug
```

## 20. `scripts/inspect_window_alignment.py`

### Purpose

Verify that context and future windows are continuous and correctly normalized.

### Required Checks

For random samples:

```text
plot last two seconds of context and first two seconds of future
plot all 12 leads
verify boundary continuity
verify sample timestamps
verify lead ordering
verify normalization mean and standard deviation
verify future R-peak indices
verify record ID and window start
```

### Required Metrics

```text
absolute boundary jump per lead
median boundary jump
99th-percentile boundary jump
context standard deviation per lead
future standard deviation per lead
percentage of near-constant windows
percentage of windows with no future R-peaks
future R-peak count distribution
```

Large discontinuities or inconsistent scaling must be fixed before further model training.

## 21. `scripts/compare_stage_checkpoints.py`

### Purpose

Compare Stage A, B, and C checkpoints on the same validation windows.

### Required Outputs

For each checkpoint, save:

```text
posterior reconstruction
deterministic prior mean forecast
128-sample prior forecast
latent trajectory plots
per-lead metric tables
uncertainty metric tables
```

Use identical:

```text
validation examples
random seed
Brownian seeds
initial-state samples
visualization limits
```

This avoids confusing checkpoint differences with sampling variation.

---

# Part III: Unified Metrics for Effective Debugging

## 22. Metric Categories

Every experiment must separate metrics into:

```text
waveform fidelity
rhythm fidelity
latent dynamics
uncertainty
representation quality
optimization health
data integrity
```

A single total loss is not sufficient for debugging.

## 23. Waveform Fidelity Metrics

Record:

```text
MSE
MAE
macro lead-wise Pearson
median lead-wise Pearson
per-lead Pearson
normalized RMSE
waveform temporal standard deviation
waveform amplitude range
derivative MAE
spectral magnitude error
```

Report separately for:

```text
all 12 leads
limb leads
precordial leads
Lead II
```

Lead II is useful for rhythm visualization but must not replace 12-lead evaluation.

## 24. Rhythm Metrics

Record:

```text
R-peak precision
R-peak recall
R-peak F1
zero-R-peak forecast percentage
R-peak count error
next-R-peak timing MAE
heart-rate MAE
RR-interval MAE
RMSSD error when enough peaks exist
```

Flat forecasts often look acceptable under MAE while failing all rhythm metrics.

## 25. Latent Dynamics Metrics

Record:

```text
initial-state mean distance
initial-state variance distance
trajectory teacher MSE
drift teacher MSE
latent temporal standard deviation
latent derivative norm
mean latent path length
prior-posterior path distance
autonomous rollout divergence
```

Define path length:

\[
L_{\text{path}}
=
\frac{1}{B}
\sum_b
\sum_t
\left\|
z_{t+1}^{(b)}
-
z_t^{(b)}
\right\|_2.
\]

A near-zero path length indicates latent stagnation.

A low teacher-forced drift loss with poor autonomous trajectory matching indicates exposure-bias failure.

## 26. Uncertainty Metrics

Record:

```text
prior log-variance statistics
posterior log-variance statistics
empirical latent sample variance
empirical waveform sample variance
90% interval width
interval coverage
sample pairwise distance
R-peak count variance
heart-rate variance
```

Coverage must be computed pointwise:

```text
fraction of ground-truth waveform values inside the 5th–95th percentile band
```

Also report that this is pointwise coverage, not coherent trajectory coverage.

A narrow interval around a flat forecast is not successful calibration.

Always report interval width together with coverage and mean forecast quality.

## 27. Representation Metrics

Record probe metrics for:

```text
next-R-peak timing
recent RR interval
future heart rate
future peak count
```

Compare:

```text
global summary
boundary token
recent summary
combined representation
```

This determines whether the context representation contains the information required by the forecasting task.

## 28. Optimization Health Metrics

Record per module:

```text
gradient norm
parameter norm
update norm
learning rate
fraction of zero gradients
fraction of non-finite gradients
```

Modules:

```text
context encoder
context pooling
posterior encoder
posterior drift
prior mean head
prior log-variance head
prior drift
decoder
rhythm head
diffusion
observation scale
```

Also record:

```text
weighted loss contribution from every loss term
ratio of each auxiliary loss to waveform loss
gradient clipping frequency
AMP overflow or skipped-step count
```

A small scalar loss can still dominate gradients, so both loss magnitude and gradient magnitude are required.

## 29. Data Integrity Metrics

Record:

```text
number of records
number of windows
windows per record
future R-peak count distribution
window boundary discontinuity
context/future mean and standard deviation
per-lead amplitude distribution
percentage of clipped values
percentage of near-constant windows
```

Verify splits by record ID.

No windows from one record may occur in multiple splits.

## 30. Standard Debug Output Format

Every debugging script will write:

```text
artifacts/debug/<script_name>/<checkpoint_name>/
```

Include:

```text
summary.json
per_sample_metrics.csv
per_lead_metrics.csv
config.yaml
selected_examples.json
plots/
```

### `summary.json`

Store aggregate metrics and configuration:

```json
{
  "checkpoint": "...",
  "forecast_horizon_seconds": 0.5,
  "num_samples": 128,
  "waveform": {},
  "rhythm": {},
  "latent": {},
  "uncertainty": {},
  "optimization": {}
}
```

### `per_sample_metrics.csv`

Include:

```text
record_id
window_start
forecast_horizon
rollout_type
mse
mae
macro_pearson
rpeak_f1
heart_rate_mae
latent_temporal_std
waveform_temporal_std
```

### `per_lead_metrics.csv`

Include:

```text
record_id
window_start
lead_name
rollout_type
mse
mae
pearson
amplitude_range
temporal_std
```

---

# Part IV: Decision Gates

## 31. Gate 0 — Data Integrity

Pass when:

```text
context and future windows are continuous
lead order is correct
normalization is context-only
R-peak annotations align with future windows
tiny-set direct model can overfit
```

## 32. Gate 1 — Direct Forecast Learnability

Pass when the direct TCN produces recognizable 12-lead ECG at 0.5 seconds.

Minimum debug target:

```text
macro Pearson >= 0.70 on validation
R-peak F1 >= 0.70
zero-R-peak forecast percentage < 20%
```

These are prototype gates, not publication targets.

## 33. Gate 2 — Deterministic Latent Prior Overfit

Pass when the deterministic latent prior overfits 32 windows at 0.5 seconds.

Target:

```text
macro Pearson >= 0.90
R-peak F1 >= 0.90
nonzero latent path length
nonconstant waveform
```

## 34. Gate 3 — Deterministic Validation Forecast

Pass when the deterministic latent prior produces recognizable 0.5-second validation forecasts.

Then extend to:

```text
1.0 second
2.0 seconds
```

Do not extend the horizon after a failed gate.

## 35. Gate 4 — Latent Utilization

Pass when decoder ablations show:

```text
shuffling latent trajectories substantially worsens waveform metrics
zeroing latent trajectories substantially worsens rhythm metrics
latent gradient sensitivity is nontrivial
context-only decoding cannot reproduce the future waveform
```

## 36. Gate 5 — Boundary Phase Representation

Pass when the combined boundary-aware representation predicts next-R-peak timing substantially better than the global summary alone.

Suggested minimum improvement:

```text
at least 20% lower next-R-peak timing MAE
```

## 37. Gate 6 — Stochastic Forecasting

Pass when:

```text
deterministic prior mean remains ECG-like
multiple samples remain ECG-like
sample trajectories are distinct
90% interval does not collapse
interval coverage improves without excessive width
R-peak timing variation is physiologically plausible
```

---

# Part V: Immediate Implementation Order

## 38. First Implementation Batch

Add:

```text
scripts/inspect_window_alignment.py
scripts/diagnose_rollout_components.py
scripts/diagnose_decoder_dependence.py
scripts/diagnose_uncertainty_sources.py
scripts/probe_context_phase.py
scripts/overfit_deterministic_prior.py
scripts/train_direct_tcn_baseline.py
scripts/compare_stage_checkpoints.py
```

Add shared utilities:

```text
src/ecg_forecast/debug/rollouts.py
src/ecg_forecast/debug/metrics.py
src/ecg_forecast/debug/checkpoint_loader.py
src/ecg_forecast/debug/reporting.py
```

## 39. Second Implementation Batch

Modify the main pipeline:

```text
fully freeze Stage B teacher
disable Stage B sampling
disable Stage B diffusion
use prior and posterior means
add autonomous trajectory matching
add initial-state mean matching
add forecast-horizon curriculum
```

Do not yet change the decoder or context architecture.

Run the diagnostic gates.

## 40. Third Implementation Batch

Based on diagnostics:

```text
add boundary-aware context representation
replace the block decoder with a temporal decoder
add explicit R-peak timing supervision
```

Retrain Stage A because the representation and decoder have changed.

Then repeat deterministic prior training.

## 41. Fourth Implementation Batch

After deterministic two-second forecasting passes:

```text
reintroduce prior variance
reintroduce bounded diffusion
reintroduce normalized KL
evaluate multisample uncertainty
```

---

# 42. Final Completion Criteria

The revised prototype is ready for stochastic research experiments when:

1. The 12-lead data pipeline passes continuity and annotation checks.
2. A direct TCN baseline predicts recognizable 0.5-second ECG.
3. The deterministic latent prior overfits a tiny dataset.
4. The deterministic latent prior generalizes at 0.5 seconds.
5. The horizon curriculum reaches two seconds.
6. Decoder ablations confirm meaningful latent utilization.
7. Boundary-aware context improves cardiac phase prediction.
8. Posterior reconstruction remains strong.
9. Prior mean forecasts are ECG-like before stochasticity is enabled.
10. Reintroduced stochastic samples remain diverse and physiologically plausible.
11. The 90% interval no longer collapses around a flat waveform.
12. All metrics, per-lead results, and diagnostic artifacts are reproducibly saved.
