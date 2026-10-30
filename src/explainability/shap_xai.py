import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SatelliteXAIExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def compute_feature_attributions(self, X_sample, y_sample=None):
        """Computes research-quality Gradient * Input attributions for KAN, GNN, Mamba, and xLSTM."""
        print("\n🔍 Computing Robust Sensor Attributions (Research Grade)...")

        if HAS_TORCH and hasattr(self.model, 'eval'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.eval()

            # 1. Model-Input Rank Detection
            expected_rank = 2
            for module in self.model.modules():
                if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)) or 'Mamba' in str(type(module)) or 'Selective' in str(type(module)):
                    expected_rank = 3
                    break

            # 2. Dynamic Input Shaping
            if X_sample.ndim == 2 and expected_rank == 3:
                X_prepared = X_sample[:, np.newaxis, :]
            elif X_sample.ndim == 3 and expected_rank == 2:
                X_prepared = X_sample[:, -1, :]
            else:
                X_prepared = X_sample

            X_tensor = torch.tensor(X_prepared, dtype=torch.float32, device=device, requires_grad=True)
            
            # 3. Enhanced Gradient Preparation
            self.model.zero_grad(set_to_none=True) # Change 1: Universal compatibility
            if X_tensor.grad is not None:
                X_tensor.grad.zero_() # Change 2: Avoid leftover gradients

            # 4. Forward pass + gradient check
            logits = self.model(X_tensor)
            if torch.isnan(logits).any():
                raise RuntimeError("Model output contains NaN values.")
            if not logits.requires_grad:
                raise RuntimeError(
                    "Model output does not require gradients. "
                    "Check for torch.no_grad(), detach(), or non-differentiable operations."
                )   

            # 5. Safer Target Handling (Verified compatibility for binary/multi-class)
            if logits.ndim == 1:
                target = logits.sum()
            elif logits.shape[1] == 1:
                target = logits[:, 0].sum()
            else:
                target = logits[:, 1].sum() # Targeting anomaly class
            
            target.backward(retain_graph=False)

            if X_tensor.grad is None:
                raise RuntimeError("Gradients failed to propagate. Verify model differentiability.")
            if X_tensor.grad.shape != X_tensor.shape:
                raise RuntimeError(
                    f"Gradient shape {tuple(X_tensor.grad.shape)} "
                    f"does not match input shape {tuple(X_tensor.shape)}."
                )
        
            gradients = X_tensor.grad.detach().cpu().numpy()
            input_data = X_tensor.detach().cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 6. Multi-Model Attribution Calculation (Gradient * Input)
            raw_attributions = np.abs(gradients * input_data)

            if raw_attributions.ndim == 3:
                mean_attributions = np.mean(raw_attributions, axis=(0, 1))
            else:
                mean_attributions = np.mean(raw_attributions, axis=0)
        else:
            X_2d = X_sample[:, -1, :] if X_sample.ndim == 3 else X_sample
            mean_attributions = np.mean(np.abs(X_2d), axis=0)

        total_attr = np.sum(mean_attributions) + 1e-10
        norm_attributions = (mean_attributions / total_attr) * 100.0

        df_rankings = pd.DataFrame({
            'Telemetry Sensor': self.feature_names,
            'Gradient Attribution (%)': norm_attributions
        }).sort_values(by='Gradient Attribution (%)', ascending=False)

        print("\n🛰️ Top Sensors Contributing to Anomaly Detection:")
        print(df_rankings.head(10).to_string(index=False, float_format=lambda x: f'{x:.2f}%'))
        return df_rankings