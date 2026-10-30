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
    class sLSTMCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(sLSTMCell, self).__init__()
            self.hidden_size = hidden_size

            self.W_i = nn.Linear(input_size, hidden_size)
            self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

            self.W_f = nn.Linear(input_size, hidden_size)
            self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

            self.W_o = nn.Linear(input_size, hidden_size)
            self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

            self.W_z = nn.Linear(input_size, hidden_size)
            self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x, states=None):
            batch_size = x.size(0)
            if states is None:
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                c = torch.zeros(batch_size, self.hidden_size, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
                m = torch.zeros(batch_size, self.hidden_size, device=x.device)
            else:
                h, c, n, m = states

            i_tilde = self.W_i(x) + self.U_i(h)
            f_tilde = self.W_f(x) + self.U_f(h)
            o_tilde = self.W_o(x) + self.U_o(h)
            z_tilde = torch.tanh(self.W_z(x) + self.U_z(h))

            m_next = torch.max(f_tilde + m, i_tilde)

            i = torch.exp(i_tilde - m_next)
            f = torch.exp(f_tilde + m - m_next)
            o = torch.sigmoid(o_tilde)

            c_next = f * c + i * z_tilde
            n_next = f * n + i
            h_next = o * (c_next / (n_next + 1e-5))

            return h_next, (h_next, c_next, n_next, m_next)

    class xLSTMClassifier(nn.Module):
        def __init__(self, in_features, hidden_dim=64, num_classes=2, num_layers=2):
            super(xLSTMClassifier, self).__init__()
            self.num_layers = num_layers
            self.cells = nn.ModuleList([
                sLSTMCell(in_features if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.SiLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            batch_size, seq_len, _ = x.size()
            layer_inputs = x
            for cell in self.cells:
                h_seq = []
                states = None
                for t in range(seq_len):
                    x_t = layer_inputs[:, t, :]
                    h_t, states = cell(x_t, states)
                    h_seq.append(h_t)
                layer_inputs = torch.stack(h_seq, dim=1)

            final_h = layer_inputs[:, -1, :]
            logits = self.classifier(final_h)
            return logits

class xLSTMNumPyFallback:
    """NumPy-native sLSTM / xLSTM Model for standalone CLI execution."""
    def __init__(self, in_features, hidden_dim=32, num_classes=2):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        np.random.seed(42)
        self.W_in = np.random.randn(in_features, hidden_dim) * 0.1
        self.W_out = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b_out = np.zeros(num_classes)

    def slstm_forward(self, X_3d):
        batch, seq, feat = X_3d.shape
        h = np.zeros((batch, self.hidden_dim))
        c = np.zeros((batch, self.hidden_dim))
        n = np.zeros((batch, self.hidden_dim))
        m = np.zeros((batch, self.hidden_dim))

        for t in range(seq):
            x_t = np.dot(X_3d[:, t, :], self.W_in)
            i_tilde = x_t
            f_tilde = x_t * 0.9
            z_tilde = np.tanh(x_t)

            m_next = np.maximum(f_tilde + m, i_tilde)
            i = np.exp(np.clip(i_tilde - m_next, -10, 10))
            f = np.exp(np.clip(f_tilde + m - m_next, -10, 10))

            c = f * c + i * z_tilde
            n = f * n + i
            h = c / (n + 1e-5)
            m = m_next

        return h

    def fit(self, X_3d, y, epochs=20, lr=0.01):
        for epoch in range(epochs):
            h = self.slstm_forward(X_3d)
            logits = np.dot(h, self.W_out) + self.b_out
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

            y_onehot = np.zeros((len(y), self.num_classes))
            y_onehot[np.arange(len(y)), y] = 1.0

            grad_logits = (probs - y_onehot) / len(y)
            self.W_out -= lr * np.dot(h.T, grad_logits)
            self.b_out -= lr * np.sum(grad_logits, axis=0)

    def predict_proba(self, X_3d):
        h = self.slstm_forward(X_3d)
        logits = np.dot(h, self.W_out) + self.b_out
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X_3d):
        return np.argmax(self.predict_proba(X_3d), axis=1)

def train_xlstm_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25, lr=0.003, batch_size=64):
    """Trains and evaluates the xLSTM (Extended LSTM) model."""
    print("\n⚡ Training xLSTM (Extended LSTM with sLSTM Memory)...")
    start_time = time.time()

    if X_train.ndim == 2:
        X_train_3d = X_train[:, None, :]
        X_val_3d = X_val[:, None, :]
        X_test_3d = X_test[:, None, :]
    else:
        X_train_3d, X_val_3d, X_test_3d = X_train, X_val, X_test

    if HAS_TORCH:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = X_train_3d.shape[2]
        model = xLSTMClassifier(in_features=in_features, hidden_dim=64, num_classes=2).to(device)

        train_x = torch.tensor(X_train_3d, dtype=torch.float32).to(device)
        train_y = torch.tensor(y_train, dtype=torch.long).to(device)
        val_x = torch.tensor(X_val_3d, dtype=torch.float32).to(device)
        val_y = torch.tensor(y_val, dtype=torch.long).to(device)
        test_x = torch.tensor(X_test_3d, dtype=torch.float32).to(device)

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
                print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(X_train_3d):.4f} | Val Acc: {val_acc*100:.2f}%")

        train_time = time.time() - start_time
        model.eval()
        inf_start = time.time()
        with torch.no_grad():
            logits = model(test_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        inf_time = (time.time() - inf_start) / len(X_test_3d) * 1000

        print(f"✅ xLSTM Training Complete in {train_time:.2f}s | Inference: {inf_time:.4f} ms/sample")
        return {
            'model_name': 'xLSTM',
            'model': model,
            'y_pred': preds,
            'y_prob': probs,
            'train_time': train_time,
            'latency_ms': inf_time
        }
    else:
        model = xLSTMNumPyFallback(in_features=X_train_3d.shape[2], hidden_dim=32, num_classes=2)
        model.fit(X_train_3d, y_train, epochs=epochs, lr=lr)

        train_time = time.time() - start_time
        inf_start = time.time()
        probs_all = model.predict_proba(X_test_3d)
        probs = probs_all[:, 1]
        preds = model.predict(X_test_3d)
        inf_time = (time.time() - inf_start) / len(X_test_3d) * 1000

        print(f"✅ xLSTM (NumPy Engine) Training Complete in {train_time:.2f}s | Inference: {inf_time:.4f} ms/sample")
        return {
            'model_name': 'xLSTM',
            'model': model,
            'y_pred': preds,
            'y_prob': probs,
            'train_time': train_time,
            'latency_ms': inf_time
        }
