import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class KANLinear(nn.Module):
        def __init__(self, in_features, out_features, num_grids=5):
            super(KANLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_grids = num_grids

            self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))

            self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_grids))
            nn.init.normal_(self.spline_weight, std=0.1)

        def chebyshev_basis(self, x):
            x_norm = torch.tanh(x)
            bases = [torch.ones_like(x_norm), x_norm]
            for k in range(2, self.num_grids):
                t_k = 2 * x_norm * bases[-1] - bases[-2]
                bases.append(t_k)
            return torch.stack(bases, dim=-1)

        def forward(self, x):
            base_output = torch.matmul(x, self.base_weight.t())
            basis = self.chebyshev_basis(x)
            spline_output = torch.einsum('bin,oin->bo', basis, self.spline_weight)
            return base_output + spline_output

    class KANClassifier(nn.Module):
        def __init__(self, in_features, hidden_dim=64, num_classes=2, num_grids=5):
            super(KANClassifier, self).__init__()
            self.kan1 = KANLinear(in_features, hidden_dim, num_grids=num_grids)
            self.act1 = nn.SiLU()
            self.kan2 = KANLinear(hidden_dim, hidden_dim // 2, num_grids=num_grids)
            self.act2 = nn.SiLU()
            self.kan3 = KANLinear(hidden_dim // 2, num_classes, num_grids=num_grids)

        def forward(self, x):
            x = self.act1(self.kan1(x))
            x = self.act2(self.kan2(x))
            x = self.kan3(x)
            return x

class KANNumPyFallback:
    """NumPy-native Kolmogorov-Arnold Network for standalone CLI execution."""
    def __init__(self, in_features, hidden_dim=32, num_classes=2, num_grids=5):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_grids = num_grids

        np.random.seed(42)
        self.W1 = np.random.randn(in_features, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(num_classes)

    def chebyshev_basis(self, X):
        X_norm = np.tanh(X)
        bases = [np.ones_like(X_norm), X_norm]
        for k in range(2, self.num_grids):
            t_k = 2 * X_norm * bases[-1] - bases[-2]
            bases.append(t_k)
        return np.stack(bases, axis=-1)

    def fit(self, X, y, epochs=20, lr=0.01):
        for epoch in range(epochs):
            # Forward pass
            basis = self.chebyshev_basis(X)
            h1 = np.tanh(np.dot(X, self.W1) + self.b1 + np.sum(basis, axis=-1)[:, :self.hidden_dim])
            logits = np.dot(h1, self.W2) + self.b2
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

            # One-hot target
            y_onehot = np.zeros((len(y), self.num_classes))
            y_onehot[np.arange(len(y)), y] = 1.0

            # Gradient step
            grad_logits = (probs - y_onehot) / len(y)
            grad_W2 = np.dot(h1.T, grad_logits)
            grad_b2 = np.sum(grad_logits, axis=0)

            grad_h1 = np.dot(grad_logits, self.W2.T) * (1.0 - h1**2)
            grad_W1 = np.dot(X.T, grad_h1)
            grad_b1 = np.sum(grad_h1, axis=0)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

    def predict_proba(self, X):
        basis = self.chebyshev_basis(X)
        h1 = np.tanh(np.dot(X, self.W1) + self.b1 + np.sum(basis, axis=-1)[:, :self.hidden_dim])
        logits = np.dot(h1, self.W2) + self.b2
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def train_kan_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25, lr=0.003, batch_size=64):
    """Trains and evaluates the Kolmogorov-Arnold Network (KAN)."""
    print("\n🧠 Training Kolmogorov-Arnold Network (KAN)...")
    start_time = time.time()

    if HAS_TORCH:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = X_train.shape[1]
        model = KANClassifier(in_features=in_features, hidden_dim=64, num_classes=2).to(device)

        train_x = torch.tensor(X_train, dtype=torch.float32).to(device)
        train_y = torch.tensor(y_train, dtype=torch.long).to(device)
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_y = torch.tensor(y_val, dtype=torch.long).to(device)
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for b_x, b_y in dataloader:
                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b_x)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_out = model(val_x)
                    val_preds = torch.argmax(val_out, dim=1)
                    val_acc = (val_preds == val_y).float().mean().item()
                print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(X_train):.4f} | Val Acc: {val_acc*100:.2f}%")

        train_time = time.time() - start_time
        model.eval()
        inf_start = time.time()
        with torch.no_grad():
            logits = model(test_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        inf_time = (time.time() - inf_start) / len(X_test) * 1000

        print(f"✅ KAN Training Complete in {train_time:.2f}s | Inference: {inf_time:.4f} ms/sample")
        return {
            'model_name': 'KAN',
            'model': model,
            'y_pred': preds,
            'y_prob': probs,
            'train_time': train_time,
            'latency_ms': inf_time
        }
    else:
        # NumPy Fallback execution
        model = KANNumPyFallback(in_features=X_train.shape[1], hidden_dim=32, num_classes=2)
        model.fit(X_train, y_train, epochs=epochs, lr=lr)

        train_time = time.time() - start_time
        inf_start = time.time()
        probs_all = model.predict_proba(X_test)
        probs = probs_all[:, 1]
        preds = model.predict(X_test)
        inf_time = (time.time() - inf_start) / len(X_test) * 1000

        print(f"✅ KAN (NumPy Engine) Training Complete in {train_time:.2f}s | Inference: {inf_time:.4f} ms/sample")
        return {
            'model_name': 'KAN',
            'model': model,
            'y_pred': preds,
            'y_prob': probs,
            'train_time': train_time,
            'latency_ms': inf_time
        }
