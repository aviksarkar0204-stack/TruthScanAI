from flask import Flask, render_template, request, jsonify
import joblib
from src.preprocessing import preprocessing

app = Flask(__name__)

svm_model = joblib.load('models/svm.pkl')
lr_model = joblib.load('models/logistic_regression.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    article = data.get('article','')
    model = data.get('model', 'svm')

    # Select the right model
    if model == 'svm':
        selected_model = svm_model
    else:
        selected_model = lr_model

    # Preprocess the raw article text
    cleaned = preprocessing(article)

    # Make a prediction 0 - fakenews, 1 - realnews
    prediction = selected_model.predict([cleaned])[0]
    is_positive = (prediction == 1)

    # Confidence scores
    if model == 'svm':
        score = selected_model.decision_function([cleaned])[0]
        real_prob = min(max(int((score + 2) / 4 * 100), 0), 100)
    else:
        proba = selected_model.predict_proba([cleaned])[0]
        real_prob = int(proba[1] * 100)  # index 1 = real class

    fake_prob = 100 - real_prob

    # Return JSON to JavaScript
    return jsonify({
        'verdict': 'real' if is_positive else 'fake',
        'real_prob': real_prob,
        'fake_prob': fake_prob
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

