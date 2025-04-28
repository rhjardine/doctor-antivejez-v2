document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURACIÓN ---
    const API_ENDPOINT = '/api/epigenetic-report/WELLMHMMD183'; // Cambia por tu endpoint real
    const PATIENT_ID = 'WELLMHMMD183'; // O obtener de la URL/autenticación

    // --- MAPEO DE TEXTO CUALITATIVO A CLASE CSS Y RANGO ---
    const qualitativeMap = {
        "Necesidad Alta": { className: "high", range: [71, 120], color: 'var(--danger-color)' },
        "Necesidad Media": { className: "medium", range: [50, 70], color: 'var(--orange-color)' },
        "Necesidad Baja": { className: "low", range: [29, 49], color: 'var(--warning-color)' },
        "Nivel Optimizado": { className: "optimized", range: [7, 28], color: 'var(--success-color)' },
    };

    // --- CHART INSTANCES ---
    let nutrientChartInstance = null;
    let exposureChartInstance = null;
    const detailChartInstances = {}; // Almacenar gráficos de indicadores

    // --- DOM ELEMENTS ---
    const patientNameEls = document.querySelectorAll('#patient-name, #intro-patient-name, #intro-hello-name');
    const patientIdEls = document.querySelectorAll('#patient-id, #intro-patient-id');
    const reportDateEl = document.getElementById('report-date');
    const catHighUl = document.getElementById('cat-high');
    const catMediumUl = document.getElementById('cat-medium');
    const catLowUl = document.getElementById('cat-low');
    const detailedSectionsContainer = document.getElementById('detailed-sections-container');
    const loadingPlaceholder = document.querySelector('.loading-placeholder');

    // --- FUNCIONES DE AYUDA ---

    // Obtener clase CSS y color basado en texto cualitativo
    function getQualitativeStyle(qualitativeText) {
        return qualitativeMap[qualitativeText] || { className: "unknown", range: [0, 0], color: 'var(--text-light)' };
    }

    // Formatear Fecha (simple)
    function formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            const [year, month, day] = dateString.split('-');
            return `${day}.${month}.${year}`;
        } catch (e) {
            return dateString; // Devolver original si falla
        }
    }

    // Crear Gráfico de Barras Resumen
    function createSummaryBarChart(canvasId, label, data) {
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return null;

        const labels = Object.keys(data);
        const values = labels.map(key => data[key].quantitative);
        const backgroundColors = labels.map(key => getQualitativeStyle(data[key].qualitative).color);

        // Destruir gráfico anterior si existe
        if (canvasId === 'nutrientsChart' && nutrientChartInstance) nutrientChartInstance.destroy();
        if (canvasId === 'exposureChart' && exposureChartInstance) exposureChartInstance.destroy();

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(c => c.replace('var(', 'rgba(').replace(')', ', 0.8)')), // Ligeramente más oscuro
                    borderWidth: 1,
                    borderRadius: 5,
                }]
            },
            options: {
                indexAxis: 'y', // Barras horizontales
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: false, // Empezar cerca del mínimo real
                        min: 0,
                        max: 120, // Escala fija
                        grid: { display: false },
                        ticks: { font: { size: 10, family: 'Poppins' }, color: 'var(--text-light)' }
                    },
                    y: {
                        grid: { display: false, drawBorder: false },
                        ticks: { font: { size: 11, family: 'Poppins', weight: '500' }, color: 'var(--text-dark-primary)' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0,0,0,0.7)',
                        titleFont: { family: 'Poppins', weight: '600' },
                        bodyFont: { family: 'Poppins' },
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.raw || 0;
                                const qualText = labels[context.dataIndex] ? data[labels[context.dataIndex]].qualitative : '';
                                return `${labels[context.dataIndex]}: ${value.toFixed(1)} (${qualText})`;
                            }
                        }
                    },
                     datalabels: { display: false } // Ocultar etiquetas dentro de barras
                }
            }
        });
        return chart;
    }

    // Crear Gráfico Gauge/Doughnut para Indicador Detallado
    function createIndicatorGaugeChart(canvasId, label, value, qualitativeText) {
         const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return;

        const style = getQualitativeStyle(qualitativeText);
        const color = style.color;
        const valueInRange = Math.max(style.range[0], Math.min(style.range[1], value)); // Limitar valor al rango
        const maxRange = 120; // Valor máximo de la escala total
        const percentage = (value / maxRange).toFixed(2); // Porcentaje sobre 120

        const data = {
            // labels: [label, 'Restante'],
            datasets: [{
                data: [valueInRange, maxRange - valueInRange], // Valor dentro del rango vs restante
                backgroundColor: [color, 'rgba(0,0,0,0.05)'], // Color del rango vs gris muy claro
                borderColor: [color, 'rgba(0,0,0,0.08)'],
                borderWidth: 0.5,
                circumference: 270,
                rotation: 225,
                cutout: '70%',
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false },
                datalabels: {
                    display: true,
                    formatter: (val, context) => {
                        if (context.dataIndex === 0) {
                            return value.toFixed(0); // Mostrar valor numérico original
                        }
                        return null;
                    },
                    color: 'var(--text-dark-primary)',
                    font: { size: 18, weight: '600', family: 'Poppins' },
                    anchor: 'center',
                    align: 'center'
                }
            }
        };

         // Destruir gráfico anterior si existe
         if (detailChartInstances[canvasId]) {
            detailChartInstances[canvasId].destroy();
         }
         detailChartInstances[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: options,
            plugins: [ChartDataLabels]
        });
    }


    // --- FUNCIÓN PRINCIPAL DE CARGA Y RENDERIZADO ---
    async function loadAndRenderReport() {
        try {
            // Simulación de fetch - Reemplazar con fetch real
            // const response = await fetch(API_ENDPOINT);
            // if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            // const data = await response.json();

             // Usar datos de ejemplo si la API no está lista
            const data = getMockData(); // Reemplazar con fetch real cuando esté listo

            if (!data || !data.patient_info || !data.summary || !data.detailed_indicators) {
                 throw new Error("Formato de datos inválido recibido de la API.");
            }

            // 1. Rellenar Información del Paciente
            const patientName = data.patient_info.name || 'N/A';
            const patientId = data.patient_info.id || 'N/A';
            const reportDate = formatDate(data.patient_info.report_date);
            patientNameEls.forEach(el => el.textContent = patientName);
            patientIdEls.forEach(el => el.textContent = patientId);
            if(reportDateEl) reportDateEl.textContent = `Fecha: ${reportDate}`;

            // 2. Crear Gráficos de Resumen
            if(data.summary.nutrient_valuation) {
               nutrientChartInstance = createSummaryBarChart('nutrientsChart', 'Valoración Nutrientes', data.summary.nutrient_valuation);
            }
             if(data.summary.exposure_load) {
               exposureChartInstance = createSummaryBarChart('exposureChart', 'Carga/Exposición', data.summary.exposure_load);
            }


            // 3. Poblar Listas de Resumen por Categoría
            const allIndicators = Object.values(data.detailed_indicators).flat();
            catHighUl.innerHTML = ''; catMediumUl.innerHTML = ''; catLowUl.innerHTML = ''; // Limpiar
            allIndicators.forEach(indicator => {
                 const style = getQualitativeStyle(indicator.qualitative);
                 const li = document.createElement('li');
                 li.textContent = indicator.name;
                 if(style.className === 'high') catHighUl.appendChild(li);
                 else if(style.className === 'medium') catMediumUl.appendChild(li);
                 else if(style.className === 'low') catLowUl.appendChild(li);
            });

            // 4. Generar Secciones Detalladas con Gráficos
            if(loadingPlaceholder) loadingPlaceholder.style.display = 'none'; // Ocultar carga
            detailedSectionsContainer.innerHTML = ''; // Limpiar contenedor

            for (const category in data.detailed_indicators) {
                 const indicators = data.detailed_indicators[category];
                 if (indicators.length === 0) continue; // Saltar categorías vacías

                 const section = document.createElement('section');
                 section.className = 'indicator-section';
                 section.id = category.toLowerCase().replace(/\s+/g, '-'); // Crear ID para navegación

                 const header = document.createElement('div');
                 header.className = 'indicator-header';
                 header.innerHTML = `<h3>${category}</h3>`; // Añadir icono si se desea
                 section.appendChild(header);

                 const grid = document.createElement('div');
                 grid.className = 'indicator-grid';

                 indicators.forEach(indicator => {
                     const card = document.createElement('div');
                     card.className = 'indicator-card';
                     const chartId = `chart-${category}-${indicator.name.replace(/\s+/g, '')}`;
                     const style = getQualitativeStyle(indicator.qualitative);

                     card.innerHTML = `
                         <div class="indicator-name">${indicator.name}</div>
                         <div class="indicator-chart-container">
                             <canvas id="${chartId}"></canvas>
                         </div>
                         <div class="indicator-qualitative qualitative-${style.className}">
                             ${indicator.qualitative}
                         </div>
                     `;
                     grid.appendChild(card);

                     // Crear el gráfico después de añadir el card al DOM (importante para obtener contexto)
                     // Usar setTimeout para asegurar que el DOM se haya actualizado
                     setTimeout(() => createIndicatorGaugeChart(chartId, indicator.name, indicator.quantitative, indicator.qualitative), 0);
                 });

                 section.appendChild(grid);
                 detailedSectionsContainer.appendChild(section);
            }

        } catch (error) {
            console.error("Error al cargar o renderizar el informe:", error);
             if(loadingPlaceholder) {
                loadingPlaceholder.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error al cargar el informe: ${error.message}`;
                loadingPlaceholder.style.color = 'var(--danger-color)';
             }
            // Mostrar un mensaje de error más prominente al usuario si es necesario
        }
    }

    // --- DATOS MOCK (Temporal - Reemplazar con llamada API) ---
    function getMockData() {
      // Simula la estructura de datos devuelta por la API
      return { /* ... (Copia la estructura JSON del paso 1 aquí) ... */ "patient_info": { "id": "WELLMHMMD183", "name": "Juan Carlos Mendez", "report_date": "2023-11-01" }, "summary": { "nutrient_valuation": { "Vitamins": {"qualitative": "Necesidad Alta", "quantitative": 85.5}, "Antioxidants": {"qualitative": "Necesidad Media", "quantitative": 62.1}, "FattyAcids": {"qualitative": "Necesidad Baja", "quantitative": 35.8}, "Minerals": {"qualitative": "Necesidad Alta", "quantitative": 98.0}, "AminoAcids": {"qualitative": "Nivel Optimizado", "quantitative": 15.3} }, "exposure_load": { "Microbiome": {"qualitative": "Necesidad Alta", "quantitative": 75.1}, "Interferences": {"qualitative": "Necesidad Media", "quantitative": 55.9}, "ToxicExposure": {"qualitative": "Necesidad Baja", "quantitative": 40.2} } }, "detailed_indicators": { "Vitamins": [ {"name": "Betaína", "qualitative": "Necesidad Media", "quantitative": 52.0}, {"name": "Biotina", "qualitative": "Nivel Optimizado", "quantitative": 25.5}, {"name": "Inositol", "qualitative": "Necesidad Alta", "quantitative": 90.1}, {"name": "Vitamina A1", "qualitative": "Necesidad Baja", "quantitative": 45.3}, {"name": "Vitamina B2", "qualitative": "Necesidad Alta", "quantitative": 72.8}, {"name": "Vitamina B9", "qualitative": "Necesidad Alta", "quantitative": 88.1}, {"name": "Vitamina B12", "qualitative": "Necesidad Media", "quantitative": 65.0} ], "Minerals": [ {"name": "Boro", "qualitative": "Necesidad Baja", "quantitative": 33.0}, {"name": "Calcio", "qualitative": "Nivel Optimizado", "quantitative": 18.0}, {"name": "Cobre", "qualitative": "Necesidad Alta", "quantitative": 71.5}, {"name": "Sodio", "qualitative": "Necesidad Baja", "quantitative": 48.8}, {"name": "Zinc", "qualitative": "Necesidad Media", "quantitative": 59.2} ], "Antioxidants": [ {"name": "Ácido Alfa-Lipoico", "qualitative": "Necesidad Alta", "quantitative": 78.0}, {"name": "Antocianinas", "qualitative": "Necesidad Media", "quantitative": 55.5}, {"name": "Polifenoles", "qualitative": "Necesidad Alta", "quantitative": 82.3}, {"name": "Superoxido Dismutasa (SOD)", "qualitative": "Necesidad Baja", "quantitative": 40.0} ], "AminoAcids": [ {"name": "Arginina", "qualitative": "Nivel Optimizado", "quantitative": 22.0}, {"name": "Ácido aspártico", "qualitative": "Necesidad Baja", "quantitative": 30.5}, {"name": "Carnosina", "qualitative": "Necesidad Media", "quantitative": 68.0}, {"name": "Ornitina", "qualitative": "Nivel Optimizado", "quantitative": 15.0}, {"name": "Valina", "qualitative": "Necesidad Baja", "quantitative": 42.0} ], "FattyAcids": [ {"name": "Ácido Alfa-Linolénico - 3 (ALA)", "qualitative": "Necesidad Media", "quantitative": 58.0}, {"name": "Ácido Araquidónico - 6 (AA)", "qualitative": "Necesidad Alta", "quantitative": 75.5}, {"name": "Ácido Docosahexaenoico - 3 (DHA)", "qualitative": "Necesidad Baja", "quantitative": 38.1} ], "Microbiome": [ {"name": "Esporas", "qualitative": "Nivel Optimizado", "quantitative": 20.0}, {"name": "Hongos", "qualitative": "Necesidad Alta", "quantitative": 95.2}, {"name": "Señal Viral", "qualitative": "Necesidad Alta", "quantitative": 80.0}, {"name": "Parásitos", "qualitative": "Necesidad Baja", "quantitative": 41.0} ] } };
    }


    // --- INICIAR CARGA ---
    loadAndRenderReport();

}); // Fin DOMContentLoaded