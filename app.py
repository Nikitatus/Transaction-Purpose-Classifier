from flask import Flask, request, jsonify
import joblib

model = joblib.load('logreg.pkl')

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

    text = data['purpose_text']
    prediction = model.predict([text])[0]

    return jsonify({ 'predicted_type': str(prediction) })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)