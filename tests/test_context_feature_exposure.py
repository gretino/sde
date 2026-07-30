import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models.context_encoder import ContextEncoder


def test_context_feature_exposure():
    cfg = ModelConfig()
    encoder = ContextEncoder(
        num_leads=cfg.num_leads,
        context_dim=cfg.context_dim,
        latent_dim=cfg.latent_dim,
    )

    b = 4
    c_wf = torch.randn(b, 500, cfg.num_leads)

    feats = encoder.encode_features(c_wf)

    assert "tokens" in feats
    assert "global" in feats
    assert "boundary" in feats
    assert "recent" in feats
    assert "dynamic" in feats

    assert feats["tokens"].shape[0] == b
    assert feats["global"].shape == (b, cfg.context_dim)
    assert feats["boundary"].shape == (b, cfg.context_dim)
    assert feats["recent"].shape == (b, cfg.context_dim)
    assert feats["dynamic"].shape == (b, cfg.context_dim)

