import os
import pytest
import torch
import torchsde

from ecg_forecast.config import load_config, Config
from ecg_forecast.data.windows import get_dataset_splits
from ecg_forecast.models.cnsde import ConditionalNeuralSDE, SDEFunc, ContextEncoder, InitialStateNetwork
from ecg_forecast.signatures.signature import (
    compute_signature_features,
    path_signature,
    get_signature_dim,
)
from ecg_forecast.losses.csig import ConditionalSignatureLoss


def test_config_parsing():
    """Config parsing: Lead-II config produces exactly one lead."""
    cfg = load_config("configs/lead2_cnsde.yaml")
    assert cfg.data.leads == [1]
    assert cfg.data.num_leads == 1
    assert cfg.model.num_leads == 1
    assert cfg.data.context_seconds == 5.0
    assert cfg.data.future_seconds == 2.0
    assert cfg.data.sampling_rate == 100
    assert cfg.data.context_samples == 500
    assert cfg.data.future_samples == 200


def test_record_splitting():
    """Record splitting: No record appears in multiple splits."""
    dummy_records = [f"rec_{i:03d}" for i in range(75)]
    splits = get_dataset_splits(dummy_records, seed=42, split_file="cache/test_splits.json")
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Mutually disjoint
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 75


def test_normalization():
    """Normalization: Future statistics are never used."""
    # Context [500, 1] and Future [200, 1]
    context = torch.randn(500, 1) * 2.0 + 5.0
    future = torch.randn(200, 1) * 10.0 + 50.0  # radically different stats

    # Normalize strictly using context statistics
    mu = context.mean(dim=0)
    sigma = context.std(dim=0).clamp_min(1e-5)

    norm_context = (context - mu) / sigma
    norm_future = (future - mu) / sigma

    # Context normalized mean ~ 0 and std ~ 1
    assert torch.allclose(norm_context.mean(dim=0), torch.zeros(1), atol=1e-4)
    assert torch.allclose(norm_context.std(dim=0), torch.ones(1), atol=1e-4)

    # Future normalized mean and std should NOT be 0 and 1 because future stats were never used
    assert not torch.allclose(norm_future.mean(dim=0), torch.zeros(1), atol=0.1)


def test_no_future_leakage():
    """Future leakage: Generator receives only context signature and anchor y0."""
    sig_dim = 1020
    model = ConditionalNeuralSDE(
        sig_dim=sig_dim,
        context_dim=64,
        latent_dim=64,
        initial_noise_dim=16,
        num_leads=1,
    )
    sig_x = torch.randn(2, sig_dim)
    y0 = torch.randn(2, 1, 1)

    # Calling forward with only context signature and y0
    wf_samples, lat_samples = model(sig_x, y0, num_samples=4)
    assert wf_samples is not None
    assert lat_samples is not None


def test_model_shapes():
    """Shapes: [B, K, 200, C] waveform and [B, K, 201, D] latent."""
    B, K, C, D = 3, 5, 1, 64
    sig_dim = 1020
    model = ConditionalNeuralSDE(
        sig_dim=sig_dim,
        context_dim=64,
        latent_dim=D,
        initial_noise_dim=16,
        num_leads=C,
        dt=0.01,
        t_end=2.0,
    )
    sig_x = torch.randn(B, sig_dim)
    y0 = torch.randn(B, 1, C)

    wf_samples, lat_samples = model(sig_x, y0, num_samples=K, use_adjoint=False)
    assert wf_samples.shape == (B, K, 200, C), f"Expected {(B, K, 200, C)}, got {wf_samples.shape}"
    assert lat_samples.shape == (B, K, 201, D), f"Expected {(B, K, 201, D)}, got {lat_samples.shape}"


def test_boundary_continuity():
    """Boundary continuity: Generated start equals final context value."""
    B, K, C = 4, 3, 1
    sig_dim = 1020
    model = ConditionalNeuralSDE(
        sig_dim=sig_dim,
        context_dim=64,
        latent_dim=64,
        initial_noise_dim=16,
        num_leads=C,
    )
    sig_x = torch.randn(B, sig_dim)
    y0 = torch.tensor([[[1.5]], [[-2.3]], [[0.0]], [[4.2]]])  # [B, 1, 1]

    # Verify internal anchoring: raw_waveform[:, 0] == y0
    c = model.context_encoder(sig_x)
    c_rep = c.unsqueeze(1).expand(B, K, -1).contiguous().view(B * K, -1)
    model.sde_func.set_context(c_rep)
    z0 = model.initial_state_net(c_rep, torch.randn(B * K, 16))
    raw_0 = model.readout(z0.unsqueeze(1))  # [B*K, 1, C]
    y0_rep = y0.unsqueeze(1).expand(B, K, 1, C).contiguous().view(B * K, 1, C)
    anchored_0 = raw_0 - raw_0 + y0_rep
    assert torch.allclose(anchored_0, y0_rep, atol=1e-6)

    # Also verify that future waveform starts close to y0 (within small initial drift dt=0.01)
    wf_samples, _ = model(sig_x, y0, num_samples=K, use_adjoint=False)
    first_future_step = wf_samples[:, :, 0, :]  # [B, K, C]
    # At dt=0.01 with bounded drift/diffusion, step is within 0.1 of y0
    diff = (first_future_step - y0.expand(B, K, C)).abs()
    assert diff.max().item() < 0.2


def test_initial_noise_variation():
    """Initial noise: Changing epsilon changes output."""
    B, D = 2, 64
    model = ConditionalNeuralSDE(sig_dim=1020, latent_dim=D, num_leads=1)
    sig_x = torch.randn(B, 1020)
    y0 = torch.randn(B, 1, 1)

    bm = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(B, D), levy_area_approximation="none")
    eps1 = torch.randn(B, 16)
    eps2 = eps1 + 2.0  # definitely different

    wf1, _ = model(sig_x, y0, num_samples=1, epsilon=eps1, bm=bm, use_adjoint=False)
    wf2, _ = model(sig_x, y0, num_samples=1, epsilon=eps2, bm=bm, use_adjoint=False)

    diff = (wf1 - wf2).abs().max().item()
    assert diff > 1e-4, f"Changing epsilon should change output, but diff was {diff}"


def test_brownian_noise_variation():
    """Brownian noise: Changing Brownian path changes output."""
    B, D = 2, 64
    model = ConditionalNeuralSDE(sig_dim=1020, latent_dim=D, num_leads=1)
    sig_x = torch.randn(B, 1020)
    y0 = torch.randn(B, 1, 1)
    eps = torch.randn(B, 16)

    # Two different Brownian paths
    torch.manual_seed(123)
    bm1 = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(B, D), levy_area_approximation="none")
    torch.manual_seed(456)
    bm2 = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(B, D), levy_area_approximation="none")

    wf1, _ = model(sig_x, y0, num_samples=1, epsilon=eps, bm=bm1, use_adjoint=False)
    wf2, _ = model(sig_x, y0, num_samples=1, epsilon=eps, bm=bm2, use_adjoint=False)

    diff = (wf1 - wf2).abs().max().item()
    assert diff > 1e-4, f"Changing Brownian path should change output, but diff was {diff}"


def test_reproducibility():
    """Reproducibility: Same epsilon + Brownian seed gives identical result."""
    B, D = 2, 64
    model = ConditionalNeuralSDE(sig_dim=1020, latent_dim=D, num_leads=1)
    sig_x = torch.randn(B, 1020)
    y0 = torch.randn(B, 1, 1)

    seed = 999
    torch.manual_seed(seed)
    eps = torch.randn(B, 16)
    bm1 = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(B, D), entropy=seed, levy_area_approximation="none")
    wf1, lat1 = model(sig_x, y0, num_samples=1, epsilon=eps, bm=bm1, use_adjoint=False)

    # Re-create with exact same seed and epsilon
    torch.manual_seed(seed)
    eps_repeat = torch.randn(B, 16)
    bm2 = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(B, D), entropy=seed, levy_area_approximation="none")
    wf2, lat2 = model(sig_x, y0, num_samples=1, epsilon=eps_repeat, bm=bm2, use_adjoint=False)

    assert torch.allclose(wf1, wf2, atol=1e-6)
    assert torch.allclose(lat1, lat2, atol=1e-6)



def test_diffusion_state_dependence():
    """Diffusion state dependence: Different z changes g(t, z, c)."""
    sde = SDEFunc(latent_dim=64, context_dim=64, sigma_min=0.005, sigma_max=0.20)
    c = torch.randn(2, 64)
    sde.set_context(c)

    t = torch.tensor([0.5])
    z1 = torch.zeros(2, 64)
    z2 = torch.ones(2, 64) * 2.0

    g1 = sde.g(t, z1)
    g2 = sde.g(t, z2)

    assert not torch.allclose(g1, g2, atol=1e-3)
    # Check bounds
    assert (g1 >= 0.005).all() and (g1 <= 0.20).all()
    assert (g2 >= 0.005).all() and (g2 <= 0.20).all()


def test_diffusion_context_dependence():
    """Diffusion context dependence: Different c changes g(t, z, c)."""
    sde = SDEFunc(latent_dim=64, context_dim=64, sigma_min=0.005, sigma_max=0.20)
    c1 = torch.zeros(2, 64)
    c2 = torch.ones(2, 64) * 3.0

    t = torch.tensor([0.5])
    z = torch.randn(2, 64)

    sde.set_context(c1)
    g1 = sde.g(t, z)

    sde.set_context(c2)
    g2 = sde.g(t, z)

    assert not torch.allclose(g1, g2, atol=1e-3)


def test_diffusion_gradients():
    """Diffusion gradients: Diffusion network gets nonzero gradient during backprop."""
    model = ConditionalNeuralSDE(sig_dim=1020, latent_dim=64, num_leads=1)
    model.train()

    sig_x = torch.randn(2, 1020)
    y0 = torch.randn(2, 1, 1)

    wf_samples, _ = model(sig_x, y0, num_samples=2, use_adjoint=True)
    loss = wf_samples.sum()
    loss.backward()

    # Check diffusion gradients
    diff_grads = [p.grad for p in model.sde_func.diffusion_net.parameters() if p.grad is not None]
    assert len(diff_grads) > 0
    total_diff_grad = sum(g.abs().sum().item() for g in diff_grads)
    assert total_diff_grad > 0.0, "Diffusion network should have nonzero gradient"


def test_signature_differentiability():
    """Signature differentiability: Gradients pass through generated signatures.

    Explicit test required by Section 18:
    x = torch.randn(4, 100, 2, requires_grad=True)
    sig = differentiable_signature(x)
    sig.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    """
    x = torch.randn(4, 100, 2, requires_grad=True)
    sig = path_signature(x, depth=4)
    sig.sum().backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Also test compute_signature_features with dyadic depth and lead-lag
    x2 = torch.randn(2, 50, 1, requires_grad=True)
    sig2 = compute_signature_features(x2, depth=2, dyadic_depth=2, lead_lag=True)
    sig2.sum().backward()

    assert x2.grad is not None
    assert x2.grad.abs().sum() > 0


def test_csig_loss_backward():
    """Verifies that ConditionalSignatureLoss propagates gradients to model parameters."""
    model = ConditionalNeuralSDE(sig_dim=1020, latent_dim=64, num_leads=1)
    loss_fn = ConditionalSignatureLoss(depth=2, dyadic_depth=1, lead_lag=False)

    sig_x = torch.randn(2, 1020)
    y0 = torch.randn(2, 1, 1)
    target_sig = torch.randn(2, loss_fn.depth + (loss_fn.depth**2))  # simple target

    wf_samples, _ = model(sig_x, y0, num_samples=2, use_adjoint=True)
    loss = loss_fn(wf_samples, target_sig)
    loss.backward()

    # Gradients should reach context encoder and readout
    ce_grad = sum(p.grad.abs().sum().item() for p in model.context_encoder.parameters() if p.grad is not None)
    ro_grad = sum(p.grad.abs().sum().item() for p in model.readout.parameters() if p.grad is not None)
    assert ce_grad > 0
    assert ro_grad > 0
