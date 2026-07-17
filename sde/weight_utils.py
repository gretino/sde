import torch
import torch.nn as nn

def load_pretrained_ecg_fm(encoder: nn.Module, weight_path: str) -> None:
    """
    Loads pre-trained feature extractor weights from ECG-FM pt file into the
    PhysiologicalEncoder's patcher (ECGFMFeatureExtractor).
    
    Args:
        encoder: The PhysiologicalEncoder instance
        weight_path: Path to the mimic_iv_ecg_finetuned.pt file
    """
    # Load the checkpoint
    checkpoint = torch.load(weight_path, map_location="cpu")
    model_state = checkpoint.get("model", checkpoint)
    
    patcher = encoder.patcher
    patcher_state_dict = patcher.state_dict()
    
    # We map 'encoder.feature_extractor.conv_layers.X.Y.weight' to 'conv_layers.X.Y.weight'
    target_state_dict = {}
    loaded_keys = []
    
    for key, val in model_state.items():
        if key.startswith("encoder.feature_extractor.conv_layers."):
            suffix = key[len("encoder.feature_extractor."):]
            if suffix in patcher_state_dict:
                # Double-check shapes match
                target_shape = patcher_state_dict[suffix].shape
                if val.shape == target_shape:
                    target_state_dict[suffix] = val
                    loaded_keys.append(suffix)
                else:
                    print(f"Warning: Shape mismatch for {key}. Pretrained: {val.shape}, Model: {target_shape}")
                    
    msg = patcher.load_state_dict(target_state_dict, strict=False)
    print(f"Successfully loaded {len(loaded_keys)} keys into the patcher: {loaded_keys}")
    print(f"Load State Dict details: {msg}")
