// Simulated AI Calculation Functions
function calculateBiologicalAge(patient) {
    let bioAge = patient.chronoAge;
    if (patient.clinicalHistory.includes("Hypertension")) bioAge += 5;
    if (patient.clinicalHistory.includes("Obesity")) bioAge += 7;
    if (patient.clinicalHistory.includes("Type 2 Diabetes")) bioAge += 7;
    if (patient.geneticMarkers.telomere_length === "short") bioAge += 3;
    else if (patient.geneticMarkers.telomere_length === "long") bioAge -= 3;
    if (patient.geneticMarkers.APOE === "ε4") bioAge += 4;
    if (patient.epigeneticData.methylation_level > 0.6) bioAge += 2;
    return Math.round(bioAge);
}

function calculateDiseaseRisks(patient) {
    const risks = { cardiovascular: 0.1, diabetes: 0.05, cancer: 0.02, alzheimers: 0.01 };
    if (patient.clinicalHistory.includes("Hypertension")) risks.cardiovascular += 0.2;
    if (patient.clinicalHistory.includes("Obesity")) { risks.cardiovascular += 0.15; risks.diabetes += 0.2; }
    if (patient.clinicalHistory.includes("Type 2 Diabetes")) risks.diabetes += 0.3;
    if (patient.geneticMarkers.APOE === "ε4") risks.alzheimers += 0.3;
    if (patient.epigeneticData.methylation_level > 0.6) risks.cancer += 0.1;
    Object.keys(risks).forEach(key => risks[key] = Math.min(risks[key], 1.0));
    return risks;
}

// Nuevo módulo predictivo de IA: Potencial de Salud vs Enfermedad
function calculateHealthPotential(patient) {
    const maxHealthScore = 100;
    let healthScore = maxHealthScore;

    // Reducir puntuación por condiciones clínicas
    if (patient.clinicalHistory.includes("Hypertension")) healthScore -= 20;
    if (patient.clinicalHistory.includes("Obesity")) healthScore -= 25;
    if (patient.clinicalHistory.includes("Type 2 Diabetes")) healthScore -= 30;

    // Ajustar por marcadores genéticos
    if (patient.geneticMarkers.telomere_length === "short") healthScore -= 15;
    else if (patient.geneticMarkers.telomere_length === "long") healthScore += 10;
    if (patient.geneticMarkers.APOE === "ε4") healthScore -= 20;

    // Ajustar por datos epigenéticos
    if (patient.epigeneticData.methylation_level > 0.6) healthScore -= 10;
    else if (patient.epigeneticData.methylation_level < 0.5) healthScore += 5;

    // Asegurar que el puntaje esté entre 0 y 100
    healthScore = Math.max(0, Math.min(healthScore, maxHealthScore));

    // Calcular riesgo de enfermedad como complemento
    const diseaseRisk = maxHealthScore - healthScore;

    return {
        healthScore: healthScore,
        diseaseRisk: diseaseScore,
        interpretation: healthScore > 70 ? "Excelente potencial de salud" :
                       healthScore > 40 ? "Potencial moderado, requiere atención" : "Alto riesgo de enfermedad"
    };
}

// Simulated API Calls
async function getAllPatients() {
    await new Promise(resolve => setTimeout(resolve, 500));
    return mockDatabase.patients;
}

async function getPatientById(id) {
    await new Promise(resolve => setTimeout(resolve, 500));
    return mockDatabase.patients.find(p => p.id === id);
}