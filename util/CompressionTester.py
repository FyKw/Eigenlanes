import os
import json
import torch
import numpy as np
from neural_compressor import quantization
from neural_compressor import pruning
from neural_compressor.utils.utility import compute_sparsity
from datetime import datetime

class CompressionTester:
    def __init__(self, model_path, test_dataloader, eval_func):
        """
        Initialize compression tester
        Args:
            model_path: Path to the original model
            test_dataloader: DataLoader for evaluation
            eval_func: Function that returns accuracy metric
        """
        self.model_path = model_path
        self.test_dataloader = test_dataloader
        self.eval_func = eval_func
        self.results = []
        self.base_model = self.load_model()
        self.base_accuracy = self.evaluate(self.base_model)
        self.base_size = self.get_model_size(self.base_model)
        
    def load_model(self):
        """Load the model from path"""
        checkpoint = torch.load(self.model_path)
        if 'model' in checkpoint:
            return checkpoint['model']
        return checkpoint

    def get_model_size(self, model):
        """Get model size in MB"""
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / (1024 * 1024)
        os.remove("temp.p")
        return size

    def evaluate(self, model):
        """Evaluate model accuracy"""
        return self.eval_func(model, self.test_dataloader)

    def test_pruning(self, ratios=[0.1, 0.3, 0.5, 0.7]):
        """Test different pruning ratios"""
        for ratio in ratios:
            # Configure pruning
            prune_conf = {
                'pruning': {
                    'magnitude': {
                        'prune_ratio': ratio
                    }
                }
            }
            
            pruner = pruning.Pruning(prune_conf)
            pruned_model = pruner(self.base_model)
            
            # Evaluate
            accuracy = self.evaluate(pruned_model)
            size = self.get_model_size(pruned_model)
            sparsity = compute_sparsity(pruned_model)
            
            result = {
                'type': 'pruning',
                'ratio': ratio,
                'accuracy': accuracy,
                'accuracy_drop': self.base_accuracy - accuracy,
                'size_mb': size,
                'size_reduction': (self.base_size - size) / self.base_size * 100,
                'sparsity': sparsity
            }
            
            self.results.append(result)
            print(f"Pruning ratio {ratio}: Accuracy={accuracy:.2f}%, Size={size:.2f}MB")

    def test_quantization(self, bits=[8, 4, 2]):
        """Test different quantization bits"""
        for bit in bits:
            # Configure quantization
            quant_conf = {
                'quantization': {
                    'approach': 'post_training_static_quant',
                    'calibration': {
                        'sampling_size': 100
                    },
                    'dtype': f'int{bit}'
                }
            }
            
            quantizer = quantization.Quantization(quant_conf)
            quantized_model = quantizer(self.base_model)
            
            # Evaluate
            accuracy = self.evaluate(quantized_model)
            size = self.get_model_size(quantized_model)
            
            result = {
                'type': 'quantization',
                'bits': bit,
                'accuracy': accuracy,
                'accuracy_drop': self.base_accuracy - accuracy,
                'size_mb': size,
                'size_reduction': (self.base_size - size) / self.base_size * 100
            }
            
            self.results.append(result)
            print(f"Quantization {bit}-bit: Accuracy={accuracy:.2f}%, Size={size:.2f}MB")

    def test_combined(self, prune_ratios=[0.3], bits=[8]):
        """Test combinations of pruning and quantization"""
        for ratio in prune_ratios:
            for bit in bits:
                # First prune
                prune_conf = {
                    'pruning': {
                        'magnitude': {
                            'prune_ratio': ratio
                        }
                    }
                }
                pruner = pruning.Pruning(prune_conf)
                pruned_model = pruner(self.base_model)
                
                # Then quantize
                quant_conf = {
                    'quantization': {
                        'approach': 'post_training_static_quant',
                        'calibration': {
                            'sampling_size': 100
                        },
                        'dtype': f'int{bit}'
                    }
                }
                quantizer = quantization.Quantization(quant_conf)
                compressed_model = quantizer(pruned_model)
                
                # Evaluate
                accuracy = self.evaluate(compressed_model)
                size = self.get_model_size(compressed_model)
                sparsity = compute_sparsity(compressed_model)
                
                result = {
                    'type': 'combined',
                    'prune_ratio': ratio,
                    'quant_bits': bit,
                    'accuracy': accuracy,
                    'accuracy_drop': self.base_accuracy - accuracy,
                    'size_mb': size,
                    'size_reduction': (self.base_size - size) / self.base_size * 100,
                    'sparsity': sparsity
                }
                
                self.results.append(result)
                print(f"Combined (P={ratio}, Q={bit}): Accuracy={accuracy:.2f}%, Size={size:.2f}MB")

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compression_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'base_model_size': self.base_size,
                'base_accuracy': self.base_accuracy,
                'results': self.results
            }, f, indent=4)
        
        print(f"Results saved to {filename}")
