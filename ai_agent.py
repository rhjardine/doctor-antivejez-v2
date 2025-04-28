# ai_agent.py
#Nota:En producción, aquí podrías integrar un modelo de ML (por ejemplo, entrenado con TensorFlow o PyTorch) para el análisis de salud integral.
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def analyze_patient_data(data):
    """
    Función dummy que procesa:
      - chronic_conditions: lista de condiciones crónicas
      - hrv: lista de mediciones del valor de variabilidad de la frecuencia cardiaca
      - psych_data: datos (por ejemplo, nivel de estrés)
    Se generan recomendaciones y un puntaje de riesgo integral.
    """
    # Datos de entrada
    chronic = data.get("chronic_conditions", [])
    hrv = data.get("hrv", [])
    psychology = data.get("psych_data", {})

    # Cálculo de score para cuidados crónicos (mayor cantidad = mayor riesgo)
    chronic_score = len(chronic) * 10

    # Promedio de VFC (suponemos que valores bajos son peores)
    if hrv:
        avg_hrv = np.mean(hrv)
        hrv_score = 100 - avg_hrv  # transformación dummy
    else:
        hrv_score = 50

    # Nivel de estrés (asumimos que escala 1 a 10)
    stress_level = psychology.get("stress_level", 5)

    # Puntaje de riesgo integral (ponderación: 50% crónicos, 30% VFC y 20% estrés)
    risk = 0.5 * chronic_score + 0.3 * hrv_score + 0.2 * (stress_level * 10)

    # Recomendaciones personalizadas en cada área
    chronic_rec = ("Revise su plan de cuidados crónicos y realice seguimientos periódicos."
                   if chronic_score > 20 else
                   "Mantenga sus hábitos actuales y realice evaluaciones anuales.")
    hrv_rec = ("Su VFC es óptima; continúe con ejercicios de respiración y relajación."
               if hrv_score < 50 else
               "La VFC indica estrés; se recomienda integración de técnicas de meditación y ejercicio moderado.")
    psych_rec = ("Niveles altos de estrés detectados; se recomienda consulta psicológica y técnicas de manejo del estrés."
                 if stress_level >= 7 else
                 "Su estado psicológico es estable; mantenga prácticas de bienestar.")

    return {
        "chronic_care": chronic_rec,
        "hrv": hrv_rec,
        "psychology": psych_rec,
        "integrated_risk_score": risk
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        recommendations = analyze_patient_data(data)
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Ejecuta en modo debug en el puerto 5000 (o el que prefieras)
    app.run(debug=True)