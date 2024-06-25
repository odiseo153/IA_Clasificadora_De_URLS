import pandas as pd
import subprocess
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_columns = [
    'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
    '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', '/AA', '/OpenAction',
    '/AcroForm', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile', '/XFA', '/Colors'
]

# Función para analizar un PDF con pdfid.py
def analyze_pdf_with_pdfid(pdf_path):
    result = subprocess.run(['python', 'pdfid.py', pdf_path], capture_output=True, text=True)
    return result.stdout

# Función para extraer características específicas del PDF
def extract_pdf_features(pdf_path):
    output = analyze_pdf_with_pdfid(pdf_path)
    lines = output.splitlines()

    pdf_data = {feature: 0 for feature in model_columns}
    for line in lines:
        for feature in model_columns:
            if line.strip().startswith(feature):
                parts = line.split()
                if len(parts) == 2:
                    key, value = parts
                    pdf_data[key] = int(value)

    return pdf_data

# Cargar el modelo guardado
xgb_model_loaded = joblib.load('pdf_model.pkl')

@app.route('/analizarpdf', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = 'temp.pdf'
    file.save(file_path)

    pdf_features = extract_pdf_features(file_path)
    pdf_features_df = pd.DataFrame([pdf_features])
    pdf_features_df = pdf_features_df.reindex(columns=model_columns, fill_value=0)

    pdf_prediction = xgb_model_loaded.predict(pdf_features_df)
    result = 'malicious' if pdf_prediction[0] else 'clean'

    return jsonify({"file": file.filename, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Cambia el puerto a 5001 o el que prefieras
