from flask import Flask, request, jsonify
import joblib

model = joblib.load('logreg.pkl')
label_encoder = joblib.load('labels_encoder.pkl')
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    """
    Expects JSON: { "purpose_text": "..." }
    Returns JSON: { "predicted_type": "..." }
    """
    data = request.get_json(force=True)
    if 'purpose_text' not in data:
        return jsonify({ 'error': 'Missing "purpose_text" in request body' }), 400

    text = data['purpose_text'].lower()
    splitted_text = text.split()
    prediction = model.predict([splitted_text])[0]
    prediction = label_encoder.inverse_transform([prediction])[0]

    return jsonify({ 'predicted_type': prediction })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
