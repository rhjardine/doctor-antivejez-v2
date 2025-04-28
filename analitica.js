document.addEventListener('DOMContentLoaded', function () {
    // Elementos del DOM
    const patientSelect = document.getElementById('patient-select');
    const analyticsContainer = document.getElementById('analytics-container');
    const noPatientMessage = document.getElementById('no-patient-selected');
    const themeToggle = document.getElementById('theme-toggle'); // Para refrescar charts

    // Selectores específicos de las secciones
    const summaryBioAgeEl = document.getElementById('summary-bio-age');
    const summaryChronoAgeEl = document.getElementById('summary-chrono-age');
    const ageDiffIndicatorEl = document.getElementById('age-difference-indicator');
    const summaryBiomarkersEl = document.getElementById('summary-biomarkers');
    const summaryRiskEl = document.getElementById('summary-risk');

    const detailedChartCanvas = document.getElementById('bioAgeVsChronoChartDetail');
    const ageComparisonGroupSelect = document.getElementById('age-comparison-group');

    const biomarkerSelect = document.getElementById('biomarker-select');
    const biomarkerTitleEl = document.getElementById('biomarker-title');
    const biomarkerLatestValueEl = document.getElementById('biomarker-latest-value');
    const biomarkerUnitEl = document.getElementById('biomarker-unit');
    const biomarkerStatusEl = document.getElementById('biomarker-status');
    const biomarkerTrendCanvas = document.getElementById('biomarkerTrendChart');
    const biomarkerComparisonTextEl = document.getElementById('biomarker-comparison-text');
    const biomarkerInfoEl = document.getElementById('biomarker-info');

    const riskCvdEl = document.getElementById('risk-cvd');
    const riskCvdComparisonEl = document.getElementById('risk-cvd-comparison');
    const riskDiabetesEl = document.getElementById('risk-diabetes');
    const riskDiabetesComparisonEl = document.getElementById('risk-diabetes-comparison');
    const riskRecommendationsEl = document.getElementById('risk-recommendations');

    const progressMetricSelect = document.getElementById('progress-metric-select');
    const progressPeriodSelect = document.getElementById('progress-period-select');
    const progressTrendCanvas = document.getElementById('progressTrendChart');
    const progressSummaryEl = document.getElementById('progress-summary');

    // Variables para almacenar instancias de gráficos
    let detailedAgeChart = null;
    let biomarkerChart = null;
    let progressChart = null;
    let currentPatientData = null; // Para almacenar datos del paciente actual

    // --- DATOS SIMULADOS ---
    // En una app real, esto vendría de llamadas API
    const patientsData = {
        p1: {
            id: 'p1', name: 'Juan Pérez', chronoAge: 55, bioAge: 58, bloodGroup: 'O+',
            biomarkers: {
                glucosa: { history: [98, 95, 96, 99, 97], unit: 'mg/dL', range: [70, 100], info: "Nivel de azúcar en sangre en ayunas." },
                hba1c: { history: [5.4, 5.3, 5.5], unit: '%', range: [4.0, 5.6], info: "Promedio de glucosa en sangre en los últimos 2-3 meses." },
                colesterol_ldl: { history: [135, 130, 128], unit: 'mg/dL', range: [0, 100], info: "Colesterol 'malo'. Idealmente bajo." },
                proteina_c_react: { history: [1.5, 1.8, 1.6], unit: 'mg/L', range: [0, 1.0], info: "Marcador de inflamación general. Idealmente bajo (<1)." },
                homocisteina: { history: [12, 11.5, 12.2], unit: 'µmol/L', range: [5, 15], info: "Aminoácido relacionado con riesgo cardiovascular." }
            },
            risks: { cvd: 12, diabetes: 8 },
            progress: { bio_age: [60, 59, 58], hscrp: [2.0, 1.8, 1.6], peso: [85, 84, 83.5], actividad: [120, 130, 150] } // Datos de progreso ejemplo
        },
        p2: {
            id: 'p2', name: 'Ana García', chronoAge: 42, bioAge: 39, bloodGroup: 'A-',
             biomarkers: {
                glucosa: { history: [88, 85, 86], unit: 'mg/dL', range: [70, 100] },
                hba1c: { history: [5.1, 5.0], unit: '%', range: [4.0, 5.6] },
                colesterol_ldl: { history: [95, 92, 90], unit: 'mg/dL', range: [0, 100] },
                proteina_c_react: { history: [0.8, 0.7, 0.6], unit: 'mg/L', range: [0, 1.0] },
                homocisteina: { history: [8, 7.8], unit: 'µmol/L', range: [5, 15] }
            },
            risks: { cvd: 5, diabetes: 3 },
             progress: { bio_age: [41, 40, 39], hscrp: [1.0, 0.8, 0.6], peso: [65, 64.5, 64], actividad: [150, 160, 180] }
        },
         p3: {
            id: 'p3', name: 'Carlos López', chronoAge: 68, bioAge: 75, bloodGroup: 'B+',
             biomarkers: {
                glucosa: { history: [110, 115, 112], unit: 'mg/dL', range: [70, 100] },
                hba1c: { history: [6.0, 6.1, 6.2], unit: '%', range: [4.0, 5.6] },
                colesterol_ldl: { history: [140, 145, 142], unit: 'mg/dL', range: [0, 100] },
                proteina_c_react: { history: [3.5, 3.8, 4.0], unit: 'mg/L', range: [0, 1.0] },
                homocisteina: { history: [16, 16.5], unit: 'µmol/L', range: [5, 15] }
            },
            risks: { cvd: 25, diabetes: 18 },
             progress: { bio_age: [74, 74.5, 75], hscrp: [3.0, 3.5, 4.0], peso: [92, 93, 93.5], actividad: [90, 80, 70] }
        },
    };
    // -------------------------

    // --- EVENT LISTENERS ---
    if (patientSelect) {
        patientSelect.addEventListener('change', handlePatientChange);
    }
    if (themeToggle) {
        themeToggle.addEventListener('click', refreshCharts); // Refresca al cambiar tema
    }
     if (ageComparisonGroupSelect) {
        ageComparisonGroupSelect.addEventListener('change', () => updateDetailedAgeChart(currentPatientData)); // Actualiza gráfico edad
    }
     if (biomarkerSelect) {
        biomarkerSelect.addEventListener('change', updateBiomarkerSection);
    }
    if (progressMetricSelect) {
         progressMetricSelect.addEventListener('change', updateProgressSection);
    }
     if (progressPeriodSelect) {
         progressPeriodSelect.addEventListener('change', updateProgressSection);
    }
    // ----------------------

    // --- LÓGICA PRINCIPAL ---

    // Inicializar estado (sin paciente seleccionado)
    showNoPatientState();

    function handlePatientChange() {
        const patientId = patientSelect.value;
        if (patientId) {
            currentPatientData = patientsData[patientId]; // Obtiene datos simulados
            if (currentPatientData) {
                showAnalyticsState();
                updateAllSections(currentPatientData);
            } else {
                showErrorState(`No se encontraron datos para el paciente ID: ${patientId}`);
            }
        } else {
            showNoPatientState();
        }
    }

    function updateAllSections(patientData) {
        console.log("Actualizando secciones para:", patientData.name);
        updateSummarySection(patientData);
        updateDetailedAgeChart(patientData); // Llama a la función que actualiza el gráfico
        updateBiomarkerSection(); // Llama a la función para biomarcadores (usará el select actual)
        updateRiskSection(patientData);
        updateProgressSection(); // Llama a la función para progreso (usará selects actuales)
        updateAIRecommendations(patientData);
    }

    function showNoPatientState() {
        if(analyticsContainer) analyticsContainer.classList.add('hidden');
        if(noPatientMessage) noPatientMessage.classList.remove('hidden');
        currentPatientData = null; // Limpia datos actuales
        // Destruye gráficos si existen
        if(detailedAgeChart) { detailedAgeChart.destroy(); detailedAgeChart = null; }
        if(biomarkerChart) { biomarkerChart.destroy(); biomarkerChart = null; }
        if(progressChart) { progressChart.destroy(); progressChart = null; }
    }

     function showErrorState(message) {
        showNoPatientState(); // Oculta contenedor principal
        if(noPatientMessage) {
            noPatientMessage.innerHTML = `<i class="fas fa-exclamation-triangle fa-3x"></i><p>${message}</p>`;
            noPatientMessage.classList.remove('hidden');
        }
    }

    function showAnalyticsState() {
        if(analyticsContainer) analyticsContainer.classList.remove('hidden');
        if(noPatientMessage) noPatientMessage.classList.add('hidden');
    }

    // --- Funciones de Actualización de Secciones ---

    function updateSummarySection(patientData) {
        if (!summaryBioAgeEl || !summaryChronoAgeEl || !ageDiffIndicatorEl || !summaryBiomarkersEl || !summaryRiskEl) return;

        summaryBioAgeEl.textContent = patientData.bioAge ?? '--';
        summaryChronoAgeEl.textContent = patientData.chronoAge ?? '--';

        // Indicador diferencia edad
        const diff = patientData.bioAge - patientData.chronoAge;
        ageDiffIndicatorEl.textContent = diff === 0 ? 'Ideal' : (diff > 0 ? `+${diff} años (Atención)` : `${diff} años (Óptimo)`);
        ageDiffIndicatorEl.className = 'age-diff ' + (diff === 0 ? 'neutral' : (diff > 0 ? 'negative' : 'positive'));

        // Resumen Biomarcadores (Ejemplo: Glucosa y hs-CRP)
        let biomarkerHTML = '<p class="loading-text">Cargando...</p>'; // Default
        if(patientData.biomarkers) {
            biomarkerHTML = '';
            const glucosaData = patientData.biomarkers.glucosa;
            const hscrpData = patientData.biomarkers.proteina_c_react;
            if(glucosaData && glucosaData.history.length > 0) {
                const latestGlucosa = glucosaData.history[glucosaData.history.length - 1];
                const status = getBiomarkerStatus(latestGlucosa, glucosaData.range);
                biomarkerHTML += `<p>Glucosa: ${latestGlucosa} ${glucosaData.unit} <span class="status-${status.level}">${status.label}</span></p>`;
            }
             if(hscrpData && hscrpData.history.length > 0) {
                const latestHscrp = hscrpData.history[hscrpData.history.length - 1];
                const status = getBiomarkerStatus(latestHscrp, hscrpData.range);
                 biomarkerHTML += `<p>hs-CRP: ${latestHscrp} ${hscrpData.unit} <span class="status-${status.level}">${status.label}</span></p>`;
            }
             if(biomarkerHTML === '') biomarkerHTML = '<p>No hay datos recientes.</p>';
        }
        summaryBiomarkersEl.innerHTML = biomarkerHTML;

        // Resumen Riesgo (Ejemplo: CVD)
        let riskHTML = '<p class="loading-text">Cargando...</p>';
         if(patientData.risks) {
            const cvdRisk = patientData.risks.cvd;
            if(cvdRisk !== undefined) {
                 const riskLevel = cvdRisk > 20 ? 'high' : cvdRisk > 10 ? 'medium' : 'low';
                 const riskLabel = cvdRisk > 20 ? 'Alto' : cvdRisk > 10 ? 'Medio' : 'Bajo';
                 riskHTML = `<p>Riesgo CV (10a): <span class="risk-level-${riskLevel}">${cvdRisk}% (${riskLabel})</span></p>`;
            } else {
                 riskHTML = '<p>No disponible</p>';
            }
        }
        summaryRiskEl.innerHTML = riskHTML;
    }

    function updateDetailedAgeChart(patientData) {
        if (!detailedChartCanvas || !patientData) return;
        const ctx = detailedChartCanvas.getContext('2d');
        if (!ctx) return;

        // Datos simulados de historial de edad biológica (para la línea)
        // En una app real, estos vendrían con el patientData
        const bioAgeHistory = patientData.progress.bio_age || [patientData.bioAge]; // Usa historial o el último valor
        const chronoAgeHistoryLabels = generateChronoAgeLabels(patientData.chronoAge, bioAgeHistory.length); // Genera etiquetas cronológicas correspondientes

        const isDarkMode = document.body.classList.contains('dark-mode');
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#ffffffcc' : '#293B64';
        const pointColorPatient = isDarkMode ? '#64b5f6' : '#23BCEF'; // Azul claro
        const lineColorPatient = pointColorPatient;
        const pointColorIdeal = isDarkMode ? '#a0a0a0' : '#888888'; // Gris
        const lineColorIdeal = pointColorIdeal;

        // --- Colores de Zonas (Ajustados ligeramente) ---
        const zonaOptimaBg = isDarkMode ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.08)';
        const zonaAtencionBg = isDarkMode ? 'rgba(255, 193, 7, 0.15)' : 'rgba(255, 193, 7, 0.1)';

        // Datos para línea ideal (y=x)
        const idealLineData = chronoAgeHistoryLabels.map(age => age);

        if (detailedAgeChart) { detailedAgeChart.destroy(); } // Destruye gráfico anterior

        detailedAgeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chronoAgeHistoryLabels, // Edades cronológicas en el eje X
                datasets: [
                    {
                        label: 'Edad Biológica (Paciente)',
                        data: bioAgeHistory, // Edades biológicas en el eje Y
                        borderColor: lineColorPatient,
                        backgroundColor: pointColorPatient,
                        tension: 0.3, fill: false, pointRadius: 5, pointHoverRadius: 7, order: 1
                    },
                    {
                        label: 'Edad Cronológica (Ideal)',
                        data: idealLineData, // Línea Y=X
                        borderColor: lineColorIdeal,
                        backgroundColor: pointColorIdeal,
                        borderDash: [5, 5], tension: 0, fill: false, pointRadius: 0, pointHoverRadius: 0, order: 2
                    }
                     // *** AQUÍ PODRÍAS AÑADIR DATASETS PARA GRUPOS DE COMPARACIÓN ***
                     // Ejemplo: Líneas punteadas para rango promedio
                    // { label: 'Rango Promedio (Sup)', data: [ ... ], borderColor: 'rgba(255, 152, 0, 0.5)', borderDash: [2, 2], pointRadius: 0, fill: '+1' },
                    // { label: 'Rango Promedio (Inf)', data: [ ... ], borderColor: 'rgba(255, 152, 0, 0.5)', borderDash: [2, 2], pointRadius: 0, fill: false }, // El 'fill: +1' rellena hasta la línea superior
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { /* ... (sin cambios) ... */ position: 'bottom', labels: { color: textColor, usePointStyle: true, padding: 20 } },
                    title: { /* ... (sin cambios) ... */ display: true, text: `Edad Biológica vs. Cronológica - ${patientData.name}`, color: textColor, font: { size: 16 } },
                    tooltip: { /* ... (sin cambios) ... */
                         mode: 'index', intersect: false, backgroundColor: isDarkMode ? 'rgba(40, 40, 40, 0.9)' : 'rgba(255, 255, 255, 0.9)', titleColor: textColor, bodyColor: textColor, borderColor: gridColor, borderWidth: 1, padding: 10, displayColors: true,
                         callbacks: { label: context => `${context.dataset.label || ''}: ${context.parsed.y !== null ? context.parsed.y.toFixed(1) + ' años' : '--'}`, title: context => `Edad Cronológica: ${context[0].label} años` }
                    },
                    annotation: { // Anotaciones para Zonas
                        annotations: {
                           zonaOptima: {
                                type: 'box', drawTime: 'beforeDatasetsDraw',
                                xScaleID: 'x', yScaleID: 'y',
                                yMin: 0,
                                yMax: (ctx) => ctx.chart.scales.x.getValueForPixel(ctx.chart.chartArea.right), // y=x
                                xMin: Math.min(...chronoAgeHistoryLabels), // Ajusta al rango de datos
                                xMax: Math.max(...chronoAgeHistoryLabels),
                                backgroundColor: zonaOptimaBg,
                                borderWidth: 0,
                                label: { content: 'Zona Óptima (Biológica ≤ Cronológica)', enabled: true, color: isDarkMode ? '#a6d8a8' : '#0a3622', position: 'start', font: { style: 'italic', size: 10 }, yAdjust: 5, xAdjust: 5 }
                            },
                            zonaAtencion: {
                                type: 'box', drawTime: 'beforeDatasetsDraw',
                                xScaleID: 'x', yScaleID: 'y',
                                yMin: (ctx) => ctx.chart.scales.x.getValueForPixel(ctx.chart.chartArea.left), // y=x
                                yMax: 120, // Límite superior del gráfico
                                xMin: Math.min(...chronoAgeHistoryLabels),
                                xMax: Math.max(...chronoAgeHistoryLabels),
                                backgroundColor: zonaAtencionBg,
                                borderWidth: 0,
                                label: { content: 'Zona de Atención (Biológica > Cronológica)', enabled: true, color: isDarkMode ? '#ffd8a1' : '#664d03', position: 'end', font: { style: 'italic', size: 10 }, yAdjust: -5, xAdjust: -5 }
                             }
                        }
                    }
                },
                scales: { // Ajusta escalas si es necesario
                    y: { title: { display: true, text: 'Edad Biológica (Años)', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor }, min: Math.min(...bioAgeHistory, ...idealLineData) - 5, max: Math.max(...bioAgeHistory, ...idealLineData) + 10 }, // Rango dinámico Y
                    x: { title: { display: true, text: 'Edad Cronológica (Años)', color: textColor }, ticks: { color: textColor }, grid: { display: false }, min: Math.min(...chronoAgeHistoryLabels) - 2, max: Math.max(...chronoAgeHistoryLabels) + 2 } // Rango dinámico X
                }
            }
        });
    }

     function updateBiomarkerSection() {
        if (!currentPatientData || !biomarkerSelect || !biomarkerTrendCanvas) return;
        const selectedBiomarkerKey = biomarkerSelect.value;
        const biomarkerData = currentPatientData.biomarkers[selectedBiomarkerKey];
        const ctx = biomarkerTrendCanvas.getContext('2d');

        if (!biomarkerData || !ctx) {
             // Mostrar estado "sin datos"
             if(biomarkerTitleEl) biomarkerTitleEl.textContent = biomarkerSelect.options[biomarkerSelect.selectedIndex].text;
             if(biomarkerLatestValueEl) biomarkerLatestValueEl.textContent = '--';
             if(biomarkerUnitEl) biomarkerUnitEl.textContent = '';
             if(biomarkerStatusEl) { biomarkerStatusEl.textContent = 'No hay datos'; biomarkerStatusEl.className = 'status'; }
             if(biomarkerComparisonTextEl) biomarkerComparisonTextEl.textContent = '--';
             if(biomarkerInfoEl) biomarkerInfoEl.textContent = '';
             if(biomarkerChart) { biomarkerChart.destroy(); biomarkerChart = null; }
            return;
        }

        const latestValue = biomarkerData.history[biomarkerData.history.length - 1];
        const status = getBiomarkerStatus(latestValue, biomarkerData.range);

        // Actualiza UI
        if(biomarkerTitleEl) biomarkerTitleEl.textContent = biomarkerSelect.options[biomarkerSelect.selectedIndex].text;
        if(biomarkerLatestValueEl) biomarkerLatestValueEl.textContent = latestValue;
        if(biomarkerUnitEl) biomarkerUnitEl.textContent = biomarkerData.unit;
        if(biomarkerStatusEl) { biomarkerStatusEl.textContent = status.label; biomarkerStatusEl.className = 'status status-' + status.level; }
        if(biomarkerComparisonTextEl) biomarkerComparisonTextEl.textContent = getBiomarkerComparisonText(latestValue, biomarkerData.range); // Texto de comparación simulado
        if(biomarkerInfoEl) biomarkerInfoEl.textContent = biomarkerData.info ?? '';

        // Actualiza Gráfico de Tendencia
        const isDarkMode = document.body.classList.contains('dark-mode');
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#ffffffcc' : '#293B64';
        const pointColor = isDarkMode ? '#81c784' : '#2e7d32'; // Verde
        const lineColor = pointColor;

        if(biomarkerChart) biomarkerChart.destroy();

        biomarkerChart = new Chart(ctx, {
            type: 'line',
            data: {
                // Genera etiquetas simples (ej. "Medición 1, 2...") o usa fechas si las tienes
                labels: biomarkerData.history.map((_, i) => `M${i+1}`),
                datasets: [{
                    label: 'Valor',
                    data: biomarkerData.history,
                    borderColor: lineColor,
                    backgroundColor: pointColor,
                    tension: 0.1, fill: false, pointRadius: 3
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: false }, tooltip: { mode: 'index', intersect: false } },
                scales: {
                     y: { ticks: { color: textColor, font: { size: 10 } }, grid: { color: gridColor }, suggestedMin: biomarkerData.range[0] * 0.8, suggestedMax: biomarkerData.range[1] * 1.2 }, // Escala Y ajustada
                     x: { ticks: { color: textColor, font: { size: 10 } }, grid: { display: false } }
                }
            }
        });
    }

    function updateRiskSection(patientData) {
        if (!riskCvdEl || !riskCvdComparisonEl || !riskDiabetesEl || !riskDiabetesComparisonEl) return;

        const cvdRisk = patientData.risks?.cvd;
        const diabetesRisk = patientData.risks?.diabetes;

        if (cvdRisk !== undefined) {
            riskCvdEl.textContent = `${cvdRisk}%`;
            riskCvdComparisonEl.textContent = getRiskComparisonText(cvdRisk, 'cvd'); // Simulado
        } else {
            riskCvdEl.textContent = '--%';
            riskCvdComparisonEl.textContent = 'No disponible';
        }

         if (diabetesRisk !== undefined) {
            riskDiabetesEl.textContent = `${diabetesRisk}%`;
            riskDiabetesComparisonEl.textContent = getRiskComparisonText(diabetesRisk, 'diabetes'); // Simulado
        } else {
            riskDiabetesEl.textContent = '--%';
            riskDiabetesComparisonEl.textContent = 'No disponible';
        }
    }

     function updateProgressSection() {
        if (!currentPatientData || !progressMetricSelect || !progressPeriodSelect || !progressTrendCanvas || !progressSummaryEl) return;

        const selectedMetric = progressMetricSelect.value;
        const selectedPeriod = progressPeriodSelect.value; // Necesitarás usar esto para filtrar datos
        const metricData = currentPatientData.progress[selectedMetric];
        const ctx = progressTrendCanvas.getContext('2d');

        if (!metricData || metricData.length < 2 || !ctx) {
            progressSummaryEl.textContent = 'No hay suficientes datos para mostrar la tendencia.';
             if(progressChart) { progressChart.destroy(); progressChart = null; }
             // Limpia el canvas si no hay datos
             ctx.clearRect(0, 0, progressTrendCanvas.width, progressTrendCanvas.height);
            return;
        }

         // --- Filtrar datos por periodo (SIMULACIÓN SIMPLE) ---
         // En una app real, necesitarías fechas asociadas a los datos
         let filteredData = metricData;
         let labels = metricData.map((_, i) => `Dato ${i+1}`); // Etiquetas genéricas
         // Aquí iría la lógica para filtrar 'filteredData' y 'labels' según 'selectedPeriod'
         // Por ejemplo, si 'selectedPeriod' es '3m' y tienes fechas, filtra los últimos 3 meses.

        // Actualiza texto resumen
        const startValue = filteredData[0];
        const endValue = filteredData[filteredData.length - 1];
        const change = endValue - startValue;
        const changePercent = ((change / startValue) * 100).toFixed(1);
        const trend = change > 0 ? 'aumento' : (change < 0 ? 'disminución' : 'estabilidad');
        const trendIcon = change > 0 ? '<i class="fas fa-arrow-trend-up" style="color: var(--color-danger);"></i>' : (change < 0 ? '<i class="fas fa-arrow-trend-down" style="color: var(--color-success);"></i>' : '<i class="fas fa-minus" style="color: var(--text-secondary);"></i>');
        // Ajustar color según si el aumento es bueno o malo para la métrica seleccionada
         progressSummaryEl.innerHTML = `Tendencia (${selectedPeriod}): ${trendIcon} ${Math.abs(change).toFixed(1)} (${Math.abs(changePercent)}%)`;

        // Actualiza gráfico
         const isDarkMode = document.body.classList.contains('dark-mode');
         const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
         const textColor = isDarkMode ? '#ffffffcc' : '#293B64';
         const pointColor = isDarkMode ? '#ffcc80' : '#ffa726'; // Naranja
         const lineColor = pointColor;

         if(progressChart) progressChart.destroy();

         progressChart = new Chart(ctx, {
             type: 'line',
             data: { labels: labels, datasets: [{ label: 'Valor', data: filteredData, borderColor: lineColor, backgroundColor: pointColor, tension: 0.3, fill: false }] },
             options: {
                 responsive: true, maintainAspectRatio: false,
                 plugins: { legend: { display: false }, title: { display: false }, tooltip: { mode: 'index', intersect: false } },
                 scales: { y: { title: { display: true, text: progressMetricSelect.options[progressMetricSelect.selectedIndex].text, color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } }, x: { title: { display: true, text: 'Tiempo', color: textColor }, ticks: { color: textColor }, grid: { display: false } } }
             }
         });
    }


    function updateAIRecommendations(patientData) {
         if (!riskRecommendationsEl) return;
         // --- SIMULACIÓN IA ---
         // Genera recomendaciones simples basadas en los datos
         let recommendations = "Recomendaciones Generales: Mantener dieta equilibrada y actividad física regular.";
         const ageDiff = patientData.bioAge - patientData.chronoAge;
         const hscrp = patientData.biomarkers?.proteina_c_react?.history.slice(-1)[0]; // Último hs-CRP
         const cvdRisk = patientData.risks?.cvd;

         if (ageDiff > 5) { recommendations += " Su edad biológica es significativamente mayor. Enfocarse en reducir factores de envejecimiento."; }
         else if (ageDiff < -3) { recommendations += " ¡Excelente! Su edad biológica es menor a la cronológica."; }

         if (hscrp && hscrp > 3) { recommendations += " Nivel de inflamación (hs-CRP) elevado, considerar dieta antiinflamatoria y manejo del estrés."; }
         else if (hscrp && hscrp < 1) { recommendations += " Nivel de inflamación bajo, ¡buen indicador!";}

         if (cvdRisk && cvdRisk > 15) { recommendations += " Riesgo cardiovascular elevado, priorizar salud cardíaca (dieta, ejercicio, no fumar)."; }

         riskRecommendationsEl.textContent = recommendations;
         // --- FIN SIMULACIÓN IA ---
    }

    // --- Funciones Helper ---
    function refreshCharts() {
        // Vuelve a dibujar todos los gráficos activos al cambiar tema
        if (currentPatientData) {
            if(detailedAgeChart) updateDetailedAgeChart(currentPatientData);
            if(biomarkerChart) updateBiomarkerSection();
            if(progressChart) updateProgressSection();
        }
    }

    function getBiomarkerStatus(value, range) {
        if (value < range[0]) return { level: 'low', label: 'Bajo' };
        if (value > range[1]) return { level: 'high', label: 'Alto' };
        return { level: 'ok', label: 'Óptimo' };
    }

    // Simula texto de comparación
    function getBiomarkerComparisonText(value, range) {
        const midPoint = (range[0] + range[1]) / 2;
        if (value > range[1] * 1.1) return "Valor significativamente superior al rango ideal.";
        if (value > range[1]) return "Valor ligeramente superior al rango ideal.";
        if (value < range[0] * 0.9) return "Valor inferior al rango ideal.";
        if (value < range[0]) return "Valor ligeramente inferior al rango ideal.";
        if (value > midPoint * 1.1) return "Dentro del rango ideal, pero en la parte alta.";
        return "Valor dentro del rango óptimo.";
    }

    function getRiskComparisonText(riskValue, riskType) {
         // Simulación muy básica
        const avgRisk = riskType === 'cvd' ? 10 : 6; // Promedios simulados
        if (riskValue > avgRisk * 1.5) return "Significativamente superior al promedio.";
        if (riskValue > avgRisk * 1.1) return "Ligeramente superior al promedio.";
        if (riskValue < avgRisk * 0.8) return "Inferior al promedio.";
        return "Dentro del promedio esperado.";
    }

    // Genera etiquetas de edad cronológica para gráficos de historial
    function generateChronoAgeLabels(currentChronoAge, historyLength) {
         // Asume que los datos son anuales hacia atrás (muy simplista)
         if (historyLength <= 1) return [currentChronoAge];
         return Array.from({ length: historyLength }, (_, i) => currentChronoAge - (historyLength - 1 - i));
     }

});

// Añade esto a tu JS principal (scripts.js) si no lo tienes,
// para manejar el menú desplegable de reportes y el toggle del sidebar/tema.
document.addEventListener('DOMContentLoaded', function () {
    // --- Sidebar Toggle ---
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.querySelector('.sidebar-toggle');
    const body = document.body;
    if (toggleBtn && sidebar && body) {
        toggleBtn.addEventListener('click', function () {
            body.classList.toggle('sidebar-expanded');
            sidebar.classList.toggle('expanded');
        });
    }

    // --- Theme Toggle ---
    // (El listener ya está en el JS específico de analitica.js,
    // pero podría estar aquí de forma centralizada también)
    // const themeToggle = document.getElementById('theme-toggle');
    // if (themeToggle) {
    //     themeToggle.addEventListener('click', () => {
    //         document.body.classList.toggle('dark-mode');
    //         // Aquí podrías necesitar llamar a una función global para refrescar TODOS los charts
    //         // if (typeof refreshAllCharts === 'function') refreshAllCharts();
    //     });
    // }


    // --- Reportes Submenu Popup ---
    const reportsItem = document.querySelector('.reports-item');
    if (reportsItem) {
        const popupSubmenu = reportsItem.querySelector('.popup-submenu');
        if (popupSubmenu) {
            reportsItem.addEventListener('mouseenter', () => { popupSubmenu.style.display = 'block'; });
            reportsItem.addEventListener('mouseleave', () => { popupSubmenu.style.display = 'none'; });
            // Para accesibilidad con teclado
            reportsItem.addEventListener('focusin', () => { popupSubmenu.style.display = 'block'; });
            reportsItem.addEventListener('focusout', (e) => {
                // Cierra solo si el foco sale del item Y del submenu
                if (!reportsItem.contains(e.relatedTarget)) {
                    popupSubmenu.style.display = 'none';
                }
            });
            // Cierra si se hace clic fuera
            document.addEventListener('click', (e) => {
                 if (!reportsItem.contains(e.target)) {
                     popupSubmenu.style.display = 'none';
                 }
            });
        }
    }
});