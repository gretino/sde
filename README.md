# ECG Natural Dynamics: Conditional Latent SDE Forecaster

Conditional Latent Stochastic Differential Equation (SDE) system for natural 12-lead ECG modeling and short-horizon future waveform forecasting.

---

## 1. Installation

Requirements: Python >= 3.9, PyTorch, `torchsde`, `wfdb`, `neurokit2`, `scipy`.

```bash
# Clone the repository
git clone https://github.com/user/sde.git
cd sde

# Install in editable mode
pip install -e .
```

---

## 2. Quick Start & Verification

### Batch Inspection & Data Verification
Inspect dataset split statistics, normalized window shapes, and model forward pass outputs:

```bash
python scripts/inspect_batch.py --config configs/debug_lead2.yaml
```

### Run Unit Test Suite
Execute the comprehensive test suite (`pytest`):

```bash
pytest tests/ -v
```

---

## 3. Training

Train the 3-stage model (Stage A Posterior Warmup $\rightarrow$ Stage B Prior Alignment $\rightarrow$ Stage C Forecast Refinement):

### Lead II Debug Configuration
```bash
python scripts/train.py --config configs/debug_lead2.yaml
```

### Full 12-Lead Production Configuration
```bash
python scripts/train.py --config configs/incart_12lead.yaml
```

Checkpoints will be saved to `checkpoints/incart_12lead/`:
- `posterior_warmup_best.pt`
- `prior_alignment_best.pt`
- `final_best.pt`

---

## 4. Evaluation

Evaluate the trained checkpoint on validation or test splits against the repeat-context baseline:

```bash
python scripts/evaluate.py --config configs/incart_12lead.yaml --checkpoint checkpoints/incart_12lead/final_best.pt --split test
```

---

## 5. Sampling Forecasts

Generate multi-sample prior forecast visualizations (16 trajectory samples with 90% prediction intervals and 12-lead grid figures):

```bash
python scripts/sample_forecasts.py --config configs/incart_12lead.yaml --checkpoint checkpoints/incart_12lead/final_best.pt --out_dir output/forecast_samples
```

---

## Repository Structure

```text
/
├── pyproject.toml
├── README.md
├── CONTEXT.md
├── docs/
│   ├── adr/
│   └── agents/
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
│       ├── config.py
│       ├── data/
│       ├── models/
│       ├── losses/
│       ├── metrics/
│       ├── training/
│       └── visualization/
└── tests/
    ├── test_dataset.py
    ├── test_model_shapes.py
    ├── test_sde_gradients.py
    ├── test_posterior_overfit.py
    ├── test_prior_forecast.py
    ├── test_sampling.py
    └── test_no_future_leakage.py
```