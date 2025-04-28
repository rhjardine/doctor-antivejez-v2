document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const navLinks = document.querySelectorAll('.nav-link');
    const contentSections = document.querySelectorAll('.content-section');
    const themeToggleBtns = document.querySelectorAll('.theme-toggle');
    const runFullAnalysisBtn = document.getElementById('runFullAnalysisBtn');
    const analysisLoader = document.getElementById('analysisLoader');
    const analysisResultsContent = document.getElementById('analysisResultsContent');
    const analysisNoData = document.getElementById('analysisNoData');
    const recommendationsLoader = document.getElementById('recommendationsLoader');
    const recommendationsContent = document.getElementById('recommendationsContent');
    const recommendationsNoData = document.getElementById('recommendationsNoData'); // Asumiendo que añades este ID en HTML
    const reportDownloadBtns = document.querySelectorAll('#reports .button'); // Botones de descarga

    // Chatbot elements
    const chatbotToggleBtn = document.getElementById('chatbotToggleBtn');
    const chatbotWindow = document.getElementById('chatbotWindow');
    const closeChatbotBtn = document.getElementById('closeChatbotBtn');
    const chatbotMessagesEl = document.getElementById('chatbotMessages');
    const chatbotInput = document.getElementById('chatbotInput');
    const chatbotSendBtn = document.getElementById('chatbotSendBtn');

    // Upload elements
    const uploadAreas = document.querySelectorAll('.upload-area');

    // Chart instances
    let longevityMarkersChart = null;
    let detailedBioAgeChart = null;
    let biomarkerOverviewChart = null; // Radar chart

    // --- State ---
    let currentSection = 'dashboard'; // Default section
    let hasUploadedData = false;      // Track if user has uploaded enough data
    let hasRunAnalysis = false;       // Track if analysis has been run

    // --- Initial Setup ---
    function initializeApp() {
        setupNavigation();
        setupTheme();
        setupUploadListeners();
        setupAnalysisButton();
        setupChatbot();
        updateUIState(); // Set initial visibility based on state
        initDashboardCharts(); // Init charts visible on dashboard
    }

    // --- Navigation ---
    function setupNavigation() {
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href')?.substring(1);
                if (targetId) {
                    setActiveSection(targetId);
                     // Scroll to top on nav change, useful for long pages
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        });
        // Set initial active section based on hash or default
        const initialHash = window.location.hash.substring(1);
        setActiveSection(initialHash || 'dashboard');
    }

    function setActiveSection(targetId) {
        contentSections.forEach(section => section.classList.toggle('active', section.id === targetId));
        navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === `#${targetId}`));
        currentSection = targetId;
        window.location.hash = targetId; // Update URL hash

        // Lazy load/init charts for the active section
        if (targetId === 'analysis' && hasRunAnalysis && !biomarkerOverviewChart) {
             initAnalysisCharts(); // Init analysis charts only if analysis was run
        }
         // Ensure dashboard chart re-inits if needed (e.g., after theme change)
         if (targetId === 'dashboard' && !longevityMarkersChart) {
            initDashboardCharts();
         }
         updateUIState(); // Update visibility based on new section and state
    }

     // --- Theme ---
    function setupTheme() {
        themeToggleBtns.forEach(btn => btn.addEventListener('click', toggleTheme));
        // Load persisted theme
        const savedTheme = localStorage.getItem('aegisTheme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
        }
        updateThemeButtonText();
    }

    function toggleTheme() {
        document.body.classList.toggle('dark-mode');
        const isDarkMode = document.body.classList.contains('dark-mode');
        localStorage.setItem('aegisTheme', isDarkMode ? 'dark' : 'light');
        updateThemeButtonText();
        // Re-render charts with new theme colors
        refreshAllCharts();
    }

     function updateThemeButtonText() {
         const isDarkMode = document.body.classList.contains('dark-mode');
         const iconClass = isDarkMode ? 'fa-sun' : 'fa-moon';
         const text = isDarkMode ? 'Modo Claro' : 'Modo Oscuro';
         themeToggleBtns.forEach(btn => {
              const iconElement = btn.querySelector('i');
              if (iconElement) {
                  iconElement.className = `fas ${iconClass}`;
                  // Check if it's the main sidebar toggle or a smaller one
                  if (btn.textContent.includes("Modo")) {
                     btn.childNodes[btn.childNodes.length -1].nodeValue = ` ${text}`; // Update text node
                  }
              } else {
                  btn.innerHTML = `<i class="fas ${iconClass}"></i> ${text}`; // Fallback
              }
         });
     }

     function getChartColors() {
         const isDarkMode = document.body.classList.contains('dark-mode');
         return {
             textColor: isDarkMode ? '#E0E0E0' : '#34495e',
             gridColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.08)',
             primary: isDarkMode ? '#64b5f6' : '#3498db', // Accent color
             secondary: isDarkMode ? '#27d7b6' : '#1abc9c', // Teal
             danger: isDarkMode ? '#e57373' : '#e74c3c',
             warning: isDarkMode ? '#ffd54f' : '#f39c12',
             optimal: isDarkMode ? '#81c784' : '#2ecc71',
             neutral: isDarkMode ? '#9E9E9E' : '#95a5a6',
             cardBg: isDarkMode ? '#2a2a2e' : '#ffffff'
         };
     }

     // --- Data Upload ---
     function setupUploadListeners() {
         uploadAreas.forEach(area => {
             const inputId = area.id.replace('upload', '').toLowerCase() + 'File';
             const statusId = area.id.replace('upload', '').toLowerCase() + 'Status';
             const fileInput = document.getElementById(inputId);
             const statusEl = document.getElementById(statusId);

             if (fileInput && statusEl) {
                 area.addEventListener('click', () => fileInput.click());
                 fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files, statusEl));

                 // Drag & Drop listeners
                 area.addEventListener('dragover', (e) => { e.preventDefault(); area.classList.add('dragging'); });
                 area.addEventListener('dragleave', () => { area.classList.remove('dragging'); });
                 area.addEventListener('drop', (e) => {
                     e.preventDefault();
                     area.classList.remove('dragging');
                     handleFileUpload(e.dataTransfer.files, statusEl);
                 });
             }
         });
     }

     function handleFileUpload(files, statusEl) {
         if (files.length > 0) {
             statusEl.textContent = 'Estado: Procesando...';
             statusEl.style.color = 'var(--status-neutral)';
             // Simulate upload & basic validation
             setTimeout(() => {
                 // TODO: Add actual upload logic here (e.g., using Fetch API)
                 // For now, just confirm visually
                 statusEl.textContent = `Estado: Cargado (${files[0].name})`;
                 statusEl.style.color = 'var(--status-optimal)';
                 hasUploadedData = true; // Mark that data is available
                 updateUIState(); // Update button states etc.
             }, 1500);
         }
     }


    // --- Analysis Logic ---
    function setupAnalysisButton() {
        if (runFullAnalysisBtn) {
            runFullAnalysisBtn.addEventListener('click', runAnalysis);
        }
    }

    function runAnalysis() {
        if (!hasUploadedData) {
            alert("Por favor, carga tus datos primero (sección 'Cargar Datos').");
            setActiveSection('data-upload'); // Redirect user
            return;
        }

        // Show loaders, hide content/placeholders
        analysisLoader?.classList.remove('hidden');
        analysisResultsContent?.classList.add('hidden');
        analysisNoData?.classList.add('hidden');
        recommendationsLoader?.classList.remove('hidden');
        recommendationsContent?.classList.add('hidden');
        recommendationsNoData?.classList.add('hidden');

        // --- SIMULATE BACKEND ANALYSIS CALL ---
        console.log("Iniciando simulación de análisis...");
        // TODO: Replace with actual fetch() call to your backend agentic system
        setTimeout(() => {
            console.log("Simulación de análisis completada.");
            if (!document.body) return; // Check if page is still loaded

            // Generate Fake Data based on simulation
            const bioAgeData = generateFakeBioAgeData();
            const longevityScore = (Math.random() * 30 + 65).toFixed(0); // Score 65-95
            const biomarkerData = generateFakeBiomarkerData(); // For radar chart
            const topBiomarkers = generateFakeTopBiomarkers(); // For list
            const nutrigenomicsText = generateFakeNutrigenomics();
            const recommendationsData = generateFakeRecommendations(bioAgeData.status);

            // --- Update Analysis Section UI ---
            const analysisBioAgeEl = document.getElementById('analysis-bio-age');
            const analysisBioAgeStatusEl = document.getElementById('analysis-bio-age-status');
            const analysisBioAgeClockEl = document.getElementById('analysis-bio-age-clock-detail');
            if(analysisBioAgeEl) analysisBioAgeEl.innerHTML = `${bioAgeData.age}<span class="unit">años</span>`;
            if(analysisBioAgeStatusEl) {
                analysisBioAgeStatusEl.textContent = bioAgeData.statusText;
                analysisBioAgeStatusEl.className = `status-indicator status-${bioAgeData.status}`;
            }
             if(analysisBioAgeClockEl) analysisBioAgeClockEl.textContent = bioAgeData.clock;


            const analysisLongevityScoreEl = document.getElementById('analysis-longevity-score-detail');
            const analysisLongevityStatusEl = document.getElementById('analysis-longevity-status-detail');
            if(analysisLongevityScoreEl) analysisLongevityScoreEl.innerHTML = `${longevityScore}<span class="unit">/100</span>`;
            if(analysisLongevityStatusEl) {
                 let longevStatus = 'neutral', longevText = 'Promedio';
                 if(longevityScore > 85) { longevStatus = 'optimal'; longevText = 'Excelente';}
                 else if (longevityScore < 70) { longevStatus = 'caution'; longevText = 'Mejorable';}
                 analysisLongevityStatusEl.textContent = longevText;
                 analysisLongevityStatusEl.className = `status-indicator status-${longevStatus}`;
            }

            updateBiomarkerOverviewChart(biomarkerData); // Update radar chart
            updateTopBiomarkersList(topBiomarkers); // Update list
            const nutrigenomicsEl = document.getElementById('nutrigenomicsSummaryDetail');
            if(nutrigenomicsEl) nutrigenomicsEl.innerHTML = `<p>${nutrigenomicsText}</p>`;

            // --- Update Recommendations Section UI ---
            updateRecommendationsUI(recommendationsData);

            // --- Update Dashboard (Optional but good UX) ---
             const dbBioAgeEl = document.getElementById('db-bio-age');
             const dbBioAgeStatusEl = document.getElementById('db-bio-age-status');
             const dbLongevityScoreEl = document.getElementById('db-longevity-score');
             const dbLongevityStatusEl = document.getElementById('db-longevity-status');
             if(dbBioAgeEl) dbBioAgeEl.innerHTML = `${bioAgeData.age}<span class="unit">años</span>`;
             if(dbBioAgeStatusEl) { dbBioAgeStatusEl.textContent = bioAgeData.statusText; dbBioAgeStatusEl.className = `status-indicator status-${bioAgeData.status}`; }
             if(dbLongevityScoreEl) dbLongevityScoreEl.innerHTML = `${longevityScore}<span class="unit">/100</span>`;
             if(dbLongevityStatusEl) { dbLongevityStatusEl.textContent = longevText; dbLongevityStatusEl.className = `status-indicator status-${longevStatus}`; }
             // Update other dashboard cards if needed

             // Hide loaders, show content
             analysisLoader?.classList.add('hidden');
             analysisResultsContent?.classList.remove('hidden');
             recommendationsLoader?.classList.add('hidden');
             recommendationsContent?.classList.remove('hidden');

             hasRunAnalysis = true; // Mark analysis as complete
             updateUIState(); // Update button states, etc.
             unlockAchievement('Primer Análisis'); // Gamification

        }, 3500); // Simulate 3.5 seconds analysis time

    }

    // --- Chart Initialization & Updates ---
    function initDashboardCharts() {
        initLongevityMarkersChart();
        // Add other dashboard charts if any
    }
    function initAnalysisCharts() {
         updateBiomarkerOverviewChart(generateFakeBiomarkerData()); // Use fake data initially
         updateDetailedBioAgeChart(generateFakeBioAgeData()); // Init detailed age chart
         // Add init for other analysis charts (Heatmap, PCA placeholders)
    }
    function refreshAllCharts() {
         // Re-init all potentially visible charts
         if (currentSection === 'dashboard' || longevityMarkersChart) {
            if(longevityMarkersChart) longevityMarkersChart.destroy();
            longevityMarkersChart = null; // Force re-init
            initLongevityMarkersChart();
         }
          if (currentSection === 'analysis' || biomarkerOverviewChart || detailedBioAgeChart) {
             if(biomarkerOverviewChart) biomarkerOverviewChart.destroy();
             if(detailedBioAgeChart) detailedBioAgeChart.destroy();
             biomarkerOverviewChart = null;
             detailedBioAgeChart = null;
             // Re-init only if analysis has run, otherwise placeholders stay
             if(hasRunAnalysis) initAnalysisCharts();
          }
    }

    function initLongevityMarkersChart() {
        const canvas = document.getElementById('longevityMarkersChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const colors = getChartColors();

        if (longevityMarkersChart) longevityMarkersChart.destroy(); // Clear previous

        longevityMarkersChart = new Chart(ctx, {
           type: 'doughnut',
           data: { /* ... (Data as before, maybe update values) ... */
                labels: ['Inflamación', 'Estrés Oxid.', 'Metabolismo', 'Reparación ADN', 'Metilación'],
                datasets: [{
                    label: 'Nivel Marcadores',
                    data: [65, 75, 80, 70, 85], // Placeholder data
                    backgroundColor: [ colors.warning+'aa', colors.danger+'aa', colors.primary+'aa', colors.secondary+'aa', colors.optimal+'aa' ], // Use theme colors with alpha
                    borderColor: colors.cardBg, borderWidth: 3, hoverOffset: 8
                }]
           },
           options: { /* ... (Options as before, using colors.textColor etc.) ... */
                responsive: true, maintainAspectRatio: false,
                plugins: {
                     legend: { position: 'bottom', labels: { color: colors.textColor, font: { size: 11 }, padding: 15, boxWidth: 12, usePointStyle: true } },
                     title: { display: true, text: 'Resumen Marcadores Clave (100 = Óptimo)', color: colors.textColor, padding: { bottom: 15 } },
                     tooltip: { backgroundColor: colors.cardBg, titleColor: colors.textColor, bodyColor: colors.textColor, borderColor: colors.borderColor, borderWidth: 1 }
                }
           }
       });
    }

    function updateDetailedBioAgeChart(bioAgeData) {
        const canvas = document.getElementById('detailedBioAgeChart');
         if (!canvas) return;
         const ctx = canvas.getContext('2d');
         if (!ctx) return;
         const colors = getChartColors();

         // Simulate some history based on current bio age for trend line
         const history = [
             bioAgeData.age + (Math.random()*4 - 2), // Add some noise
             bioAgeData.age + (Math.random()*2 - 1),
             bioAgeData.age
         ].map(a => Math.max(18, Math.round(a))); // Ensure age > 18
          const chronoLabels = generateChronoAgeLabels(50, history.length); // Assume chrono age 50 for labels

         if (detailedBioAgeChart) detailedBioAgeChart.destroy();

         detailedBioAgeChart = new Chart(ctx, {
             type: 'line',
             data: {
                 labels: chronoLabels,
                 datasets: [
                     { label: 'Edad Biológica', data: history, borderColor: colors.primary, backgroundColor: colors.primary, tension: 0.3, fill: false, pointRadius: 4 },
                     { label: 'Edad Cronológica', data: chronoLabels, borderColor: colors.neutral, backgroundColor: colors.neutral, borderDash: [5, 5], tension: 0, pointRadius: 0 }
                 ]
             },
             options: {
                 responsive: true, maintainAspectRatio: false,
                 plugins: {
                    legend: { position: 'bottom', labels: { color: colors.textColor } },
                    title: { display: true, text: 'Tendencia Edad Biológica vs. Cronológica', color: colors.textColor },
                    tooltip: { /* ... tooltip options ... */ },
                     annotation: { /* ... annotation options (as before, using colors.optimal etc.) ... */
                         annotations: {
                           zonaOptima: { type: 'box', drawTime: 'beforeDatasetsDraw', xScaleID: 'x', yScaleID: 'y', yMin: 0, yMax: (ctx) => ctx.chart.scales.x.getPixelForValue(Math.max(...chronoLabels)), xMin: Math.min(...chronoLabels), xMax: Math.max(...chronoLabels), backgroundColor: colors.optimal+'1a', borderWidth: 0, label: { /* ... */ } },
                            zonaAtencion: { type: 'box', drawTime: 'beforeDatasetsDraw', xScaleID: 'x', yScaleID: 'y', yMin: (ctx) => ctx.chart.scales.x.getPixelForValue(Math.min(...chronoLabels)), yMax: Math.max(...history) + 10, xMin: Math.min(...chronoLabels), xMax: Math.max(...chronoLabels), backgroundColor: colors.warning+'1a', borderWidth: 0, label: { /* ... */ } }
                         }
                     }
                 },
                 scales: {
                     y: { title: { display: true, text: 'Edad (Años)', color: colors.textColor }, ticks: { color: colors.textColor }, grid: { color: colors.gridColor } },
                     x: { title: { display: true, text: 'Edad Cronológica (Años)', color: colors.textColor }, ticks: { color: colors.textColor }, grid: { display: false } }
                 }
             }
         });
     }

    function updateBiomarkerOverviewChart(biomarkerData) {
         const canvas = document.getElementById('biomarkerOverviewChart');
         if (!canvas) return;
         const ctx = canvas.getContext('2d');
         if (!ctx) return;
         const colors = getChartColors();

         if(biomarkerOverviewChart) biomarkerOverviewChart.destroy();

         biomarkerOverviewChart = new Chart(ctx, {
             type: 'radar',
             data: {
                 labels: ['Glucosa', 'Inflamación', 'Lípidos', 'Hígado', 'Riñón', 'Estrés Oxid.'],
                 datasets: [{
                     label: 'Perfil Biomarcadores',
                     data: biomarkerData,
                     fill: true,
                     backgroundColor: colors.primary + '33', // Primary con alpha
                     borderColor: colors.primary,
                     pointBackgroundColor: colors.primary,
                     pointBorderColor: colors.cardBg,
                     pointHoverBackgroundColor: colors.cardBg,
                     pointHoverBorderColor: colors.primary
                 }]
             },
             options: {
                 responsive: true, maintainAspectRatio: false,
                 scales: {
                     r: { angleLines: { color: colors.gridColor }, grid: { color: colors.gridColor }, pointLabels: { color: colors.textColor, font: { size: 11 } }, suggestedMin: 0, suggestedMax: 100, ticks: { display: false } } // Oculta ticks radiales
                 },
                 plugins: { legend: { display: false }, title: { display: false }, tooltip: { /* ... */ } }
             }
         });
     }

     function updateTopBiomarkersList(topBiomarkers) {
         const listEl = document.getElementById('topBiomarkersList');
         if (!listEl) return;
         listEl.innerHTML = ''; // Clear previous
         if (!topBiomarkers || topBiomarkers.length === 0) {
             listEl.innerHTML = '<p class="report-placeholder">No hay datos destacados.</p>';
             return;
         }
          const ul = document.createElement('ul');
          ul.style.listStyle = 'none';
          topBiomarkers.forEach(b => {
               const li = document.createElement('li');
               li.style.marginBottom = '8px';
               li.style.fontSize = '0.9em';
               const statusClass = `status-${b.status}`; // e.g., status-optimal
               li.innerHTML = `
                  <span style="font-weight: 500;">${b.name}:</span>
                  <span style="font-weight: 600; margin-left: 5px;">${b.value} ${b.unit}</span>
                  <span class="${statusClass}" style="margin-left: 8px; font-weight: 500;">(${b.statusText})</span>
               `;
               ul.appendChild(li);
          });
          listEl.appendChild(ul);
     }

     // --- Recommendation UI Update (Adaptada) ---
     function updateRecommendationsUI(recommendations) {
         const map = {
             nutrition: document.getElementById('nutritionRecs')?.querySelector('ul'),
             exercise: document.getElementById('exerciseRecs')?.querySelector('ul'),
             supplement: document.getElementById('supplementRecs')?.querySelector('ul'),
             wellness: document.getElementById('wellnessRecs')?.querySelector('ul'),
             therapy: document.getElementById('therapyRecs')?.querySelector('ul'), // Añadido
         };
          // Clear previous & add placeholders
         Object.keys(map).forEach(key => {
            const ul = map[key];
            if (ul) ul.innerHTML = `<li class="placeholder-rec">Generando...</li>`;
         });

         if (!recommendations || recommendations.length === 0) {
            Object.values(map).forEach(ul => { if(ul) ul.innerHTML = '<li class="placeholder-rec">No hay recomendaciones específicas por ahora.</li>'; });
            return;
         }

         // Group recommendations
         const groupedRecs = {};
         recommendations.forEach(rec => {
             if (!groupedRecs[rec.category]) groupedRecs[rec.category] = [];
             groupedRecs[rec.category].push(rec);
         });

          // Populate lists
         Object.keys(map).forEach(key => {
            const ul = map[key];
            if (ul) {
                ul.innerHTML = ''; // Clear placeholder
                const recsForKey = groupedRecs[key] || [];
                if (recsForKey.length === 0) {
                     ul.innerHTML = '<li class="placeholder-rec" style="font-style: italic;">Sin recomendaciones específicas aquí.</li>';
                } else {
                    recsForKey.forEach(rec => {
                        const li = document.createElement('li');
                        const iconClass = getRecommendationIcon(rec.category);
                        li.innerHTML = `<i class="fas ${iconClass}"></i> <span>${rec.text}</span>`;
                        ul.appendChild(li);
                    });
                }
            }
         });
     }
     // (getRecommendationIcon function sin cambios)
     function getRecommendationIcon(category) { switch(category) { case 'nutrition': return 'fa-carrot'; case 'exercise': return 'fa-person-running'; case 'supplement': return 'fa-pills'; case 'wellness': return 'fa-spa'; case 'therapy': return 'fa-hand-holding-medical'; default: return 'fa-check'; } }

    // --- Chatbot Logic ---
    function setupChatbot() {
        chatbotToggleBtn?.addEventListener('click', () => {
            chatbotWindow?.classList.toggle('hidden');
            updateChatbotIcon();
        });
        closeChatbotBtn?.addEventListener('click', () => {
             chatbotWindow?.classList.add('hidden');
             updateChatbotIcon();
        });
        chatbotSendBtn?.addEventListener('click', handleChatSend);
        chatbotInput?.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleChatSend(); });
    }

    function updateChatbotIcon() {
         if(!chatbotToggleBtn || !chatbotWindow) return;
         const isHidden = chatbotWindow.classList.contains('hidden');
         chatbotToggleBtn.innerHTML = isHidden ? '<i class="fas fa-comment-dots"></i>' : '<i class="fas fa-times"></i>';
    }

    function handleChatSend() {
         const message = chatbotInput.value.trim();
         if (!message || !chatbotMessagesEl || !chatbotInput) return;
         appendChatMessage(message, 'user');
         chatbotInput.value = '';
         chatbotInput.disabled = true; // Disable input while waiting
         chatbotSendBtn.disabled = true;

         appendChatMessage('...', 'agent'); // Typing indicator

         // Simulate API call to agent
         // TODO: Replace with actual fetch call
         setTimeout(() => {
             if (!document.body) return; // Check if page still loaded
             const response = generateFakeChatResponse(message);
             // Remove typing indicator
              const typingIndicator = chatbotMessagesEl.querySelector('.agent-message:last-child');
              if (typingIndicator && typingIndicator.textContent === '...') {
                  typingIndicator.remove();
              }
             appendChatMessage(response, 'agent');
             chatbotInput.disabled = false; // Re-enable input
             chatbotSendBtn.disabled = false;
             chatbotInput.focus();
         }, 1200 + Math.random() * 800); // Simulate variable response time
    }

    function appendChatMessage(text, sender) {
         if(!chatbotMessagesEl) return;
         const msgDiv = document.createElement('div');
         msgDiv.classList.add('chat-message', `${sender}-message`);
         msgDiv.textContent = text;
         chatbotMessagesEl.appendChild(msgDiv);
         // Scroll to bottom smoothly
         chatbotMessagesEl.scrollTo({ top: chatbotMessagesEl.scrollHeight, behavior: 'smooth' });
    }

    // --- UI State Management ---
    function updateUIState() {
        // Show/hide sections based on whether data is uploaded/analysis run
        const showAnalysis = hasUploadedData;
        const showRecommendations = hasRunAnalysis;
        const showReports = hasRunAnalysis;

        // Toggle visibility of sections (using placeholders if no data/analysis)
        document.getElementById('analysisNoData')?.classList.toggle('hidden', showAnalysis);
        document.getElementById('analysisResultsContent')?.classList.toggle('hidden', !showAnalysis || !hasRunAnalysis);
        analysisLoader?.classList.add('hidden'); // Ensure loader is hidden initially

        document.getElementById('recommendationsNoData')?.classList.toggle('hidden', showRecommendations);
        document.getElementById('recommendationsContent')?.classList.toggle('hidden', !showRecommendations);
        recommendationsLoader?.classList.add('hidden');

        // Enable/disable report buttons
        reportDownloadBtns.forEach(btn => btn.disabled = !showReports);
        document.querySelector('#reports .report-placeholder')?.classList.toggle('hidden', showReports);

        // Enable/disable analysis button
        if(runFullAnalysisBtn) runFullAnalysisBtn.disabled = !hasUploadedData;

    }

     // --- Gamification (Simple) ---
     const achievements = new Set(); // Keep track of unlocked achievements
     function unlockAchievement(name) {
        if (achievements.has(name)) return; // Already unlocked
        const badges = document.querySelectorAll('.achievement-badge');
        badges.forEach(badge => {
            if (badge.querySelector('p').textContent === name) {
                 badge.classList.add('unlocked');
                 achievements.add(name);
                 // Optional: Show a small notification/toast
                 console.log(`Achievement Unlocked: ${name}`);
            }
        });
     }

    // --- Fake Data Generation Functions (Keep for simulation) ---
    function generateFakeBioAgeData() { const c=50; const r=(Math.random()-0.5)*10; const a=Math.max(25,Math.min(95,Math.round(c+r))); const d=a-c; let s='neutral', st='Normal'; if(d<=-4){s='optimal';st='Óptima';} else if(d>=5){s='alert';st='Elevada';} else if(d>1){s='caution';st='Moderada';} return {age:a,status:s,statusText:st, clock:Math.random()>0.5?'PhenoAge*':'Horvath*'}; }
    function generateFakeBiomarkerData() { return [ Math.random()*40+60, Math.random()*40+50, Math.random()*30+70, Math.random()*20+80, Math.random()*30+65, Math.random()*40+55 ]; }
    function generateFakeTopBiomarkers() { return [ { name: 'hs-CRP', value: (Math.random()*3 + 0.5).toFixed(1), unit: 'mg/L', status: Math.random() > 0.6 ? 'high' : 'optimal', statusText: Math.random() > 0.6 ? 'Alto' : 'Óptimo' }, { name: 'Glicemia Ayunas', value: (Math.random()*20 + 80).toFixed(0), unit: 'mg/dL', status: Math.random() > 0.8 ? 'caution' : 'optimal', statusText: Math.random() > 0.8 ? 'Moderado' : 'Óptimo' }, { name: 'Vitamina D', value: (Math.random()*30 + 25).toFixed(0), unit: 'ng/mL', status: Math.random() > 0.5 ? 'low' : 'optimal', statusText: Math.random() > 0.5 ? 'Bajo' : 'Óptimo' } ]; }
    function generateFakeNutrigenomics() { return Math.random() > 0.5 ? "Metabolismo rápido de cafeína detectado. Respuesta normal a grasas saturadas. Considerar optimizar ingesta de Colina (gen PEMT*)." : "Sensibilidad a histaminas sugerida (gen DAO*). Metabolismo lento de grasas poliinsaturadas. Priorizar Omega-3 de origen marino."; }
    function generateFakeRecommendations(status) { const r=[]; if(status==='optimal')r.push({category:'wellness',text:'¡Excelente! Mantén tus hábitos.'}); r.push({category:'nutrition',text: Math.random()>0.5?'Incrementa vegetales crucíferos.':'Asegura ingesta de fibra prebiótica.'}); r.push({category:'exercise',text: Math.random()>0.5?'Prueba entrenamiento interválico (HIIT) 1-2x/sem.':'Añade ejercicios de movilidad articular.'}); if(status==='alert'||status==='caution'){r.push({category:'wellness',text:'Implementa rutina de relajación antes de dormir.'}); r.push({category:'nutrition',text:'Reduce carbohidratos refinados y azúcares.'}); r.push({category:'supplement',text: Math.random()>0.5?'Evalúa suplementar con CoQ10 con tu médico.':'Considera Magnesio Bisglicinato nocturno.'}); } r.push({category:'therapy', text: Math.random()>0.5 ? 'Considerar terapia de luz roja para recuperación.' : 'Explorar beneficios de sauna infrarrojo.'}); return r; }
    function generateFakeChatResponse(input) { const l=input.toLowerCase(); if(l.includes("edad")) return `Tu edad biológica es ${document.getElementById('analysis-bio-age')?.textContent||'--'}. Compara con tu edad cronológica para ver tu ritmo de envejecimiento.`; if(l.includes("recomienda")||l.includes("hacer")) return "Basado en tu análisis, enfócate en mejorar [Área simulada, ej: la calidad de tu sueño y reducir la inflamación]. ¿Quieres detalles sobre cómo hacerlo?"; if(l.includes("gen ") || l.includes("nutrigen")) return "Tu perfil genético sugiere [Insight simulado, ej: una menor eficiencia en la detoxificación fase II]. Alimentos como brócoli y ajo pueden ayudar. Consulta los detalles en la sección de Análisis."; return "Entendido. Puedo buscar información sobre biomarcadores, recomendaciones o tu plan. ¿Qué te gustaría saber?"; }

    // --- Initialize App ---
    initializeApp();

});