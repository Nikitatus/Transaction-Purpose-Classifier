# Transaction Purpose Classifier

This repository provides tools to preprocess transaction purpose data, train and evaluate machine learning models, and serve the final Logistic Regression model via a Flask-based REST API.

---

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [REST API](#rest-api)
6. [Sample Requests](#sample-requests)

---

## Setup and Installation

1. **Clone the repository**

   ```bash
   git clone <https://github.com/Nikitatus/Transaction-Purpose-Classifier>
   cd <Transaction-Purpose-Classifier>
   ```
   
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preprocessing

All preprocessing steps are implemented in the Jupyter notebook `Transaction Purpose Classifier.ipynb`. 
The dataset consists from `data.csv` and `dataCategory.py` files where the latter is used to generate labels.

1. **Load raw data**

   * Data is loaded from a CSV and `CategorizeData` class from `dataCategory.py` is used to generate category labels.
2. **Text cleaning**

   * Lowercasing, punctuation removal and tokenisation.
3. **Feature extraction**

   * Convert cleaned text to one-hot using plain `MultiLabelBinarizer`.
4. **Train-test split**

   * Split the dataset into training and validation sets (80/20).

**To run preprocessing**:

In the notebook UI, run all cells under "Data Preprocessing" section.

---

## Model Training

The notebook also contains cells to train various models. By default, it trains and compares:

* Logistic Regression
* Random Forest
* XGBoost

**To train (or retrain) models**:

Run cells under "Model Training", after training, export the best pipeline:
```bash
import joblib
joblib.dump(best_pipeline, 'model_name.pkl')
```

Make sure `model_name.pkl` is saved in the project root.

---

## Model Evaluation

Evaluation metrics and their results are in the notebook:

* Accuracy, precision, recall for each class

**To evaluate**:

Run cells under "Evaluation" and review printed metrics.

---

## REST API

The file `app.py` implements a minimal Flask API that loads `logreg.pkl` and exposes a `/classify` endpoint.

## Spin up the server

```bash
# Ensure model_name.pkl is in the same directory as app.py
python app.py
```

By default, the API listens on `http://0.0.0.0:5000`.

---

## Sample Requests

Use `curl` or any HTTP client:

```bash
curl -X POST http://127.0.0.1:5000/classify \
     -H "Content-Type: application/json" \
     -d '{"purpose_text": "Rent payment for August"}'
```

Expected response:

```json
{ "predicted_type": "housing_rent" }
```

---

## LLM approach

One of the options is to choose moderately sized pre‑trained transformer like 'DistilBERT', despite its small size (compared to LLMs) it retains strong language understanding, requiring only a few gigabytes of RAM and completing a fine‑tuning run in minutes on a single GPU.

The tranformer outputs a vector summarizing texts, then we will place a small feed-forward layer on top for classification. During fine-tuning the whole model and the last layer are adjusted.
