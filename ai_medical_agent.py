from flask import Flask, request, jsonify
from google.cloud import aiplatform, storage
import os

app = Flask(__name__)

# Configuración de Vertex AI
PROJECT_ID = "tu-proyecto-id"  # Reemplaza con tu ID de proyecto
LOCATION = "us-central1"
ENDPOINT_ID = "tu-endpoint-id"  # Reemplaza con tu Endpoint ID

aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

# Configuración de Cloud Storage
BUCKET_NAME = "tu-bucket-name"  # Crea un bucket en Google Cloud Storage y usa su nombre
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "Mensaje vacío"}), 400

    try:
        # Llamada a Vertex AI para procesar el mensaje
        response = endpoint.predict(instances=[{"content": message}])
        ai_response = response.predictions[0] if response.predictions else "No se obtuvo respuesta."
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        # Subir archivo a Cloud Storage
        blob = bucket.blob(f"documents/{file.filename}")
        blob.upload_from_file(file)
        
        # Aquí podrías procesar el archivo con Vertex AI (e.g., extraer texto y analizarlo)
        # Por simplicidad, devolvemos un mensaje simulado
        return jsonify({"message": f"Documento {file.filename} subido exitosamente. Análisis pendiente."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)