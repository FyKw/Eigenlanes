import torch
import os
from collections import OrderedDict


class ModelPruner:
    def __init__(self, model_path, save_dir="pruned_models"):
        """
        Initialize the pruner with model path and save directory
        Args:
            model_path: Path to the .pth model file
            save_dir: Directory to save pruned models
        """
        if not model_path or not isinstance(model_path, str):
            raise ValueError("model_path must be a valid string")

        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load_model(self):
        """Load the model state dict"""
        try:
            print(f"Attempting to load model from: {self.model_path}")
            print(f"File size: {os.path.getsize(self.model_path)} bytes")

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print(f"Type of loaded object: {type(checkpoint)}")

            # Handle different model saving formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    print("Found 'model' key in loaded dictionary")
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    print("Found 'state_dict' key in loaded dictionary")
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    print("Found 'model_state_dict' key in loaded dictionary")
                    state_dict = checkpoint['model_state_dict']
                else:
                    raise KeyError("Could not find model weights in checkpoint")

            # Print some information about the state dict
            print(f"Number of keys in state_dict: {len(state_dict)}")
            if len(state_dict) > 0:
                print("First few keys:", list(state_dict.keys())[:3])
                # Print shape of first tensor
                first_key = next(iter(state_dict))
                if isinstance(state_dict[first_key], torch.Tensor):
                    print(f"Shape of first tensor ({first_key}): {state_dict[first_key].shape}")

            return state_dict

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def prune_weights(self, state_dict, prune_ratio):
        """
        Prune weights based on magnitude
        Args:
            state_dict: Model state dictionary
            prune_ratio: Percentage of weights to prune (0.0 to 1.0)
        Returns:
            Pruned state dictionary
        """
        pruned_state_dict = OrderedDict()

        for key, weights in state_dict.items():
            if isinstance(weights, torch.Tensor):
                # Only prune conv and linear layers
                if len(weights.shape) in [2, 4]:  # Linear or Conv layers
                    # Calculate threshold
                    abs_weights = torch.abs(weights)
                    threshold = torch.quantile(abs_weights, prune_ratio)

                    # Create mask for pruning
                    mask = torch.where(abs_weights > threshold, 1.0, 0.0)

                    # Apply mask
                    pruned_weights = weights * mask
                    pruned_state_dict[key] = pruned_weights
                else:
                    # Keep other layers unchanged
                    pruned_state_dict[key] = weights
            else:
                pruned_state_dict[key] = weights

        return pruned_state_dict

    def get_model_size(self, state_dict):
        """Calculate model size in MB"""
        total_params = 0
        for weights in state_dict.values():
            if isinstance(weights, torch.Tensor):
                total_params += weights.numel()

        return (total_params * 4) / (1024 * 1024)  # Size in MB

    def get_sparsity(self, state_dict):
        """Calculate model sparsity (percentage of zero weights)"""
        total_elements = 0
        zero_elements = 0

        for weights in state_dict.values():
            if isinstance(weights, torch.Tensor):
                total_elements += weights.numel()
                zero_elements += torch.sum(weights == 0).item()

        return (zero_elements / total_elements) * 100 if total_elements > 0 else 0

    def prune_and_save(self, prune_ratios):
        """
        Prune model with different ratios and save results
        Args:
            prune_ratios: List of pruning ratios to try
        """

        # give me some meta
        # print(f"making .md file with info about the model")
        # collect_some_data(
        original_state_dict = self.load_model()
        original_size = self.get_model_size(original_state_dict)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Original sparsity: {self.get_sparsity(original_state_dict):.2f}%")

        for ratio in prune_ratios:
            print(f"\nPruning with ratio {ratio:.2f}")

            # Prune weights
            pruned_state_dict = self.prune_weights(original_state_dict, ratio)

            # Calculate metrics
            pruned_size = self.get_model_size(pruned_state_dict)
            sparsity = self.get_sparsity(pruned_state_dict)

            print(f"Pruned model size: {pruned_size:.2f} MB")
            print(f"Sparsity: {sparsity:.2f}%")

            # Avoid division by zero
            if original_size > 0:
                size_reduction = (original_size - pruned_size) / original_size * 100
                print(f"Size reduction: {size_reduction:.2f}%")
            else:
                print("Cannot calculate size reduction: original model size is 0")

            # Save pruned model
            model_name = os.path.basename(self.model_path)
            save_path = os.path.join(self.save_dir, f"pruned_{ratio}_{model_name}")
            torch.save(pruned_state_dict, save_path)
            print(f"Saved pruned model to: {save_path}")


if __name__ == "__main__":
    model_path = "/pretrained/tusimple/pruned_checkpoint_tusimple_res_18"
    pruner = ModelPruner(model_path)

    # Try different pruning ratios
    prune_ratios = [0.3]
    pruner.prune_and_save(prune_ratios)
