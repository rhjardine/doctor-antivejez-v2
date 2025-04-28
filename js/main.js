// Load Data
async function loadData() {
    const patients = await getAllPatients();
    if (patients.length === 0) {
        document.getElementById('patient-count').textContent = 0;
        document.getElementById('new-records').textContent = 0;
        document.getElementById('bio-age-gauge').style.setProperty('--gauge-value', '0%');
        document.getElementById('bio-age-gauge').setAttribute('data-value', 0);
        return;
    }

    const bioAges = patients.map(p => calculateBiologicalAge(p));
    const avgBioAge = bioAges.reduce((sum, age) => sum + age, 0) / bioAges.length;
    document.getElementById('patient-count').textContent = patients.length;
    const oneMonthAgo = new Date();
    oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
    const newRecords = patients.filter(p => new Date(p.lastVisit) > oneMonthAgo).length;
    document.getElementById('new-records').textContent = newRecords;
    const gauge = document.getElementById('bio-age-gauge');
    gauge.style.setProperty('--gauge-value', `${(avgBioAge / 100) * 100}%`);
    gauge.setAttribute('data-value', Math.round(avgBioAge));

    const recentPatients = patients.slice(0, 5).map(p => ({
        id: p.id,
        name: p.name,
        chronoAge: p.chronoAge,
        bioAge: calculateBiologicalAge(p),
        lastVisit: p.lastVisit
    }));
    const tbody = document.getElementById('recent-patients-body');
    tbody.innerHTML = recentPatients.map(p => `
        <tr data-id="${p.id}">
            <td>${p.name}</td>
            <td>${p.chronoAge}</td>
            <td>${p.bioAge}</td>
            <td>${p.lastVisit}</td>
        </tr>
    `).join('');

    tbody.addEventListener('click', (event) => {
        const row = event.target.closest('tr');
        if (row) {
            const id = row.dataset.id;
            if (id) selectPatient(parseInt(id));
        }
    });

    if (patients.length > 0) selectPatient(patients[0].id);

    const ageGroups = {
        '30-40': patients.filter(p => p.chronoAge >= 30 && p.chronoAge < 40),
        '40-50': patients.filter(p => p.chronoAge >= 40 && p.chronoAge < 50),
        '50-60': patients.filter(p => p.chronoAge >= 50 && p.chronoAge < 60),
        '60+': patients.filter(p => p.chronoAge >= 60)
    };
    const groupData = Object.keys(ageGroups).map(group => {
        const groupPatients = ageGroups[group];
        const avgBioAge = groupPatients.length > 0 ? 
            groupPatients.map(p => calculateBiologicalAge(p)).reduce((sum, age) => sum + age, 0) / groupPatients.length : 0;
        return { group, avgBioAge };
    });
    const groupComparisonData = {
        labels: groupData.map(d => d.group),
        datasets: [{ label: 'Edad Biológica Promedio', data: groupData.map(d => d.avgBioAge), backgroundColor: ['#00c4cc', '#007b8f', '#e0f7fa', '#b2ebf2'] }]
    };
    new Chart(document.getElementById('groupComparisonChart'), { type: 'bar', data: groupComparisonData, options: { responsive: true } });
}

// Select Patient Function
async function selectPatient(id) {
    const patient = await getPatientById(id);
    if (patient) {
        document.getElementById('patient-name').textContent = patient.name;
        document.getElementById('chrono-age').textContent = patient.chronoAge;
        const bioAge = calculateBiologicalAge(patient);
        document.getElementById('bio-age').textContent = bioAge;

        const bioAgeData = {
            labels: patient.biologicalAgeHistory.map(h => h.date),
            datasets: [{ label: 'Edad Biológica', data: patient.biologicalAgeHistory.map(h => h.age), borderColor: '#00c4cc', fill: false }]
        };
        const bioAgeChart = Chart.getChart('bioAgeChart');
        if (bioAgeChart) bioAgeChart.destroy();
        new Chart(document.getElementById('bioAgeChart'), { type: 'line', data: bioAgeData, options: { responsive: true } });

        const risks = calculateDiseaseRisks(patient);
        document.getElementById('progress-cardio').value = risks.cardiovascular * 100;
        document.getElementById('risk-cardio').textContent = (risks.cardiovascular * 100).toFixed(1) + '%';
        document.getElementById('progress-diabetes').value = risks.diabetes * 100;
        document.getElementById('risk-diabetes').textContent = (risks.diabetes * 100).toFixed(1) + '%';
        document.getElementById('progress-cancer').value = risks.cancer * 100;
        document.getElementById('risk-cancer').textContent = (risks.cancer * 100).toFixed(1) + '%';
        document.getElementById('progress-alzheimers').value = risks.alzheimers * 100;
        document.getElementById('risk-alzheimers').textContent = (risks.alzheimers * 100).toFixed(1) + '%';

        const ageDifference = bioAge - patient.chronoAge;
        const differenceText = ageDifference > 0 ? `${ageDifference} años mayor` : `${-ageDifference} años menor`;
        const cardioRisk = risks.cardiovascular * 100;
        const averageCardioRisk = 10;
        const riskComparison = cardioRisk > averageCardioRisk ? 'mayor' : 'menor';
        document.getElementById('insight-text').textContent = `Basado en sus marcadores genéticos y historial clínico, su edad biológica se estima en ${bioAge} años, lo que es ${differenceText} que su edad cronológica. Su riesgo de enfermedad cardiovascular es ${cardioRisk.toFixed(1)}%, lo cual es ${riskComparison} que el promedio.`;

        // Añadir potencial de salud vs enfermedad
        const healthPotential = calculateHealthPotential(patient);
        document.getElementById('health-potential').textContent = `Potencial de Salud: ${healthPotential.healthScore}%, Riesgo de Enfermedad: ${healthPotential.diseaseRisk}%, Interpretación: ${healthPotential.interpretation}`;
    }
}

// Funciones del Modal
function showHistoriasModal(patientId) {
    const patient = mockDatabase.patients.find(p => p.id === patientId);
    if (!patient) return;

    document.getElementById('historias-modal').style.display = 'block';
    document.getElementById('prev-diseases').textContent = patient.clinicalHistory.length > 0 ? patient.clinicalHistory.join(', ') : 'Sin registro';
    document.getElementById('treatments-surgeries').textContent = patient.clinicalHistory.includes("Hypertension") ? 'Tratamiento con antihipertensivos' : 'Sin registro';
    document.getElementById('treatment-plan').textContent = generateTreatmentPlan(patient);
    const alerts = generateAlerts(patient);
    document.getElementById('alerts').innerHTML = alerts.map(alert => `<li>${alert}</li>`).join('');
}

function closeModal() {
    document.getElementById('historias-modal').style.display = 'none';
}

function saveNotes() {
    const notes = document.getElementById('doctor-notes').value;
    alert('Notas guardadas: ' + notes);
}

function generateTreatmentPlan(patient) {
    const bioAge = calculateBiologicalAge(patient);
    const ageDiff = bioAge - patient.chronoAge;
    if (ageDiff > 5) {
        return 'Dieta antienvejecimiento, ejercicio moderado diario, suplementos antioxidantes.';
    } else {
        return 'Mantener estilo de vida actual, chequeo anual recomendado.';
    }
}

function generateAlerts(patient) {
    const risks = calculateDiseaseRisks(patient);
    const alerts = [];
    if (risks.cardiovascular > 0.3) alerts.push('Alerta: Riesgo cardiovascular elevado.');
    if (new Date(patient.lastVisit) < new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)) {
        alerts.push('Recordatorio: Programar chequeo mensual.');
    }
    return alerts.length > 0 ? alerts : ['Sin alertas actuales.'];
}

// Theme Toggle
document.getElementById('theme-toggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.querySelectorAll('.card, #disease-risks, #ai-insights, .patient-details, #historias-modal > div').forEach(el => el.classList.toggle('dark-mode'));
    document.querySelector('.sidebar').classList.toggle('dark-mode');
    document.querySelector('.recent-patients').classList.toggle('dark-mode');
});

// Búsqueda
document.querySelector('header input').addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    const rows = document.querySelectorAll('#recent-patients-body tr');
    rows.forEach(row => {
        const name = row.querySelector('td:first-child').textContent.toLowerCase();
        row.style.display = name.includes(searchTerm) ? '' : 'none';
    });
});

// Asociar botón "Historias"
document.querySelector('.sidebar-item:first-child').addEventListener('click', () => showHistoriasModal(1));

// Load Data on Page Load
loadData();