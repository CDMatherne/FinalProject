"""
Model Factory
Creates and loads models from configuration
"""
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .architectures.hybrid_model import HybridLSTMTransformer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    input_size: Optional[int] = None
) -> HybridLSTMTransformer:
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary (optional)
        config_path: Path to configuration file (optional)
        input_size: Number of input features (required if not in config)
    
    Returns:
        Initialized model
    """
    # Load config if path provided
    if config_path:
        config = load_config(config_path)
    
    if config is None:
        raise ValueError("Either config or config_path must be provided")
    
    # Get model config
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # Determine input size
    if input_size is None:
        # Try to infer from config or use default
        # Default: LAT, LON, SOG, COG, Heading = 5 features
        # Plus temporal features, vessel features, etc.
        input_size = model_config.get('input_size', 10)  # Default estimate
    
    # LSTM parameters
    lstm_config = model_config.get('lstm', {})
    lstm_hidden_size = lstm_config.get('hidden_size', 128)
    lstm_num_layers = lstm_config.get('num_layers', 2)
    lstm_dropout = lstm_config.get('dropout', 0.2)
    lstm_bidirectional = lstm_config.get('bidirectional', False)
    
    # Transformer parameters
    transformer_config = model_config.get('transformer', {})
    transformer_hidden_size = transformer_config.get('hidden_size', 256)
    transformer_num_layers = transformer_config.get('num_layers', 4)
    transformer_num_heads = transformer_config.get('num_heads', 8)
    transformer_feedforward_size = transformer_config.get('feedforward_size', 1024)
    transformer_dropout = transformer_config.get('dropout', 0.1)
    
    # Output parameters
    output_config = model_config.get('output', {})
    prediction_horizon = output_config.get('prediction_horizon', 48)
    uncertainty_method = output_config.get('uncertainty_method', 'variational')
    
    # Fusion size
    fusion_hidden_size = model_config.get('fusion_hidden_size', 256)
    
    # Create model
    model = HybridLSTMTransformer(
        input_size=input_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        lstm_bidirectional=lstm_bidirectional,
        transformer_hidden_size=transformer_hidden_size,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_feedforward_size=transformer_feedforward_size,
        transformer_dropout=transformer_dropout,
        fusion_hidden_size=fusion_hidden_size,
        prediction_horizon=prediction_horizon,
        uncertainty_method=uncertainty_method,
        include_speed_course=True
    )
    
    return model


def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    device: str = 'cpu'
) -> HybridLSTMTransformer:
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        config_path: Path to configuration file (optional)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model
    """
    # Load config if provided
    config = None
    if config_path:
        config = load_config(config_path)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint or config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    elif config:
        model_config = config.get('model', {})
    else:
        raise ValueError("Model configuration not found in checkpoint or config file")
    
    # Determine input size - try multiple methods
    input_size = None
    
    # Method 1: Check if explicitly stored in checkpoint
    if 'input_size' in checkpoint:
        input_size = checkpoint['input_size']
        logger.info(f"Found input_size={input_size} in checkpoint")
    
    # Method 2: Infer from checkpoint weights (most reliable)
    if input_size is None:
        # Get state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Try to infer from LSTM input layer weight
        # The LSTM input layer weight has shape [4*hidden_size, input_size] for each direction
        # For bidirectional: [8*hidden_size, input_size], for unidirectional: [4*hidden_size, input_size]
        lstm_key = 'lstm_encoder.lstm.weight_ih_l0'
        if lstm_key in state_dict:
            lstm_weight_shape = state_dict[lstm_key].shape
            # LSTM weight_ih has shape [4*hidden_size, input_size]
            # So input_size = weight.shape[1]
            inferred_input_size = lstm_weight_shape[1]
            input_size = inferred_input_size
            logger.info(f"Inferred input_size={input_size} from LSTM weight shape {lstm_weight_shape}")
        
        # Alternative: Try transformer input projection
        if input_size is None:
            transformer_key = 'transformer_encoder.input_projection.weight'
            if transformer_key in state_dict:
                transformer_weight_shape = state_dict[transformer_key].shape
                # Transformer input_projection has shape [hidden_size, input_size]
                inferred_input_size = transformer_weight_shape[1]
                input_size = inferred_input_size
                logger.info(f"Inferred input_size={input_size} from transformer weight shape {transformer_weight_shape}")
    
    # Method 3: Fall back to config or default
    if input_size is None:
        input_size = model_config.get('input_size', 5)  # Default to 5 (LAT, LON, SOG, COG, Heading)
        logger.warning(f"Could not infer input_size, using config/default: {input_size}")
    
    logger.info(f"Using input_size={input_size} for model creation")
    
    # Create model
    model = create_model(config=config, input_size=input_size)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model(
    model: HybridLSTMTransformer,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    input_size: Optional[int] = None,
    **kwargs
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        config: Configuration dictionary (optional)
        input_size: Input size (optional)
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size or model.input_size,
        'config': config,
        **kwargs
    }
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

