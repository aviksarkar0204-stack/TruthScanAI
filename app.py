from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import torch
import torch.nn as nn
from src.preprocessing import preprocessing, dl_preprocessing

app = Flask(__name__)

# ── Load classical ML models ──
svm_model = joblib.load('models/svm.pkl')
lr_model = joblib.load('models/logistic_regression.pkl')
svm_retrained = joblib.load('models/svm_retrained.pkl')
lr_retrained = joblib.load('models/lr_retrained.pkl')

# ── Load vocabulary ──
with open('models/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# ── LSTM model definition ──
class FakeGuardLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.sigmoid(self.fc(out))
        return out

# ── Load LSTM weights ──
VOCAB_SIZE = len(word2idx)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = FakeGuardLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
lstm_model.load_state_dict(torch.load('models/fakeguard_lstm.pth', map_location=device))
lstm_model.to(device)
lstm_model.eval()

# ── LSTM inference helper ──
MAX_LEN = 500

def encode_and_pad(tokens, word2idx, max_len):
    encoded = [word2idx.get(word, 1) for word in tokens]
    if len(encoded) >= max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [0] * (max_len - len(encoded))
    return encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    article = data.get('article', '')
    model = data.get('model', 'svm')

    if model == 'lstm':
        # DL preprocessing
        tokens = dl_preprocessing(article)
        encoded = encode_and_pad(tokens, word2idx, MAX_LEN)
        tensor = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            output = lstm_model(tensor)
            prob = output.item()

        is_positive = prob >= 0.5
        real_prob = int(prob * 100)
        fake_prob = 100 - real_prob

    else:
        # Classical ML preprocessing
        if model == 'svm':
            selected_model = svm_model
        elif model == 'lr':
            selected_model = lr_model
        elif model == 'svm_retrained':
            selected_model = svm_retrained
        elif model == 'lr_retrained':
            selected_model = lr_retrained

        cleaned = preprocessing(article)
        prediction = selected_model.predict([cleaned])[0]
        is_positive = (prediction == 1)

        if model == 'svm':
            score = selected_model.decision_function([cleaned])[0]
            real_prob = min(max(int((score + 2) / 4 * 100), 0), 100)
        else:
            proba = selected_model.predict_proba([cleaned])[0]
            real_prob = int(proba[1] * 100)

        fake_prob = 100 - real_prob

    return jsonify({
        'verdict': 'real' if is_positive else 'fake',
        'real_prob': real_prob,
        'fake_prob': fake_prob
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
