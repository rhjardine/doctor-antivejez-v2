// js/dashboard.js - Lógica Específica del Dashboard

document.addEventListener('DOMContentLoaded', () => {
    console.log("Initializing Dashboard specific logic...");

    // Check if we are actually on the dashboard page (e.g., by checking for a specific element)
    if (!document.querySelector('.assistant-container')) {
        console.log("Not on Dashboard page, skipping dashboard specific JS.");
        return; // Exit if not on the dashboard
    }

    // --- Constants and State ---
    const API_BASE_URL = 'http://localhost:5000/api'; // Or get from global config
    let currentAnalysis = null;
    let currentFileId = null;
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

    // --- Element Selectors ---
    const patientNameEl = document.getElementById('patient-name');
    const patientDetailsEl = document.getElementById('patient-details');
    const biologicalAgeEl = document.getElementById('biological-age');
    const healthScoreEl = document.getElementById('health-score');
    const lastCheckupEl = document.getElementById('last-checkup');
    const assistantTabs = document.querySelectorAll('#assistant-tabs .assistant-tab'); // Scope to container
    const tabContents = document.querySelectorAll('.assistant-content > .tab-content');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatContainer = document.querySelector('#chat-tab .chat-container');
    const voiceButton = document.getElementById('voice-button');
    const attachButton = document.getElementById('attach-button');
    const fileInput = document.getElementById('file-input');
    // Document tab elements
    const uploadArea = document.getElementById('upload-area');
    const browseFilesBtn = document.getElementById('browse-files');
    const docFileInput = document.getElementById('document-file-input');
    const docPreview = document.getElementById('document-preview');
    const docNameEl = document.getElementById('document-name');
    const docContentEl = document.getElementById('document-content');
    const analyzeDocBtn = document.getElementById('analyze-document-btn');
    const docAnalysisEl = document.getElementById('document-analysis');
    const analysisContentEl = document.getElementById('analysis-content');
    const integrateBtn = document.getElementById('integrate-analysis-btn');
    const docRecordsContainer = document.getElementById('document-records');
    // ... Add selectors for other elements as needed ...


    // --- Functions ---

    // Load Patient Data (Example)
    function loadPatientData() {
        // Replace with actual API call
        const mockPatient={name:"Isabel Romero",age:58,gender:"Female",id:"458912",biologicalAge:52.3,trend:-5.7,healthScore:84,lastCheckup:"2023-04-12"};
        if(patientNameEl) patientNameEl.textContent = mockPatient.name;
        if(patientDetailsEl) patientDetailsEl.textContent = `${mockPatient.age} years • ${mockPatient.gender} • ID: ${mockPatient.id}`;
        if(biologicalAgeEl) biologicalAgeEl.innerHTML = `${mockPatient.biologicalAge} <span class="trend">(${mockPatient.trend > 0 ? '+' : ''}${mockPatient.trend})</span>`;
        if(healthScoreEl) healthScoreEl.textContent = `${mockPatient.healthScore}/100`;
        if(lastCheckupEl) lastCheckupEl.textContent = mockPatient.lastCheckup ? new Date(mockPatient.lastCheckup + 'T00:00:00').toLocaleDateString('es-ES',{day:'2-digit',month:'2-digit',year:'numeric'}) : 'N/A';
        // Load initial documents (example)
        loadDocuments();
        console.log("Patient data loaded (mock).");
    }

    // Load Documents (Example)
    function loadDocuments() {
        // Replace with actual API call
        const mockDocs = [
            { id: 'doc1', name: 'Analitica_Sangre_Jun23.pdf', type: 'PDF', size: 1200000, uploadDate: '2023-06-15T10:00:00Z' },
            { id: 'doc2', name: 'Informe_Genetico_v2.pdf', type: 'PDF', size: 2500000, uploadDate: '2023-05-20T14:30:00Z', analysis: {type: 'Genomic', results: [{gene: 'APOE', variant: 'e3/e4', interpretation: 'Riesgo moderado Alzheimer.'}]} }
        ];
        renderDocumentRecords(mockDocs);
    }

     function renderDocumentRecords(documents) {
         if (!docRecordsContainer) return;
         docRecordsContainer.innerHTML = '<h3 class="upload-text" style="text-align: left; margin-bottom: 1rem;">Documentos Analizados Previamente</h3>'; // Reset header
         if (!documents || documents.length === 0) {
             docRecordsContainer.innerHTML += '<p style="text-align:center; color: var(--text-light); margin-top: 1rem;">No hay documentos analizados.</p>';
             return;
         }
         documents.forEach(doc => {
             const record = document.createElement('div');
             record.className = 'document-record';
             record.dataset.fileId = doc.id; // Store ID for later use
             // Determine badge class (simplified)
             let badgeClass = 'document-badge';
             if (doc.type?.toLowerCase().includes('genomic')) badgeClass += ' genomic';
             else if (doc.type?.toLowerCase().includes('lab')) badgeClass += ' lab';
             else if (doc.type?.toLowerCase().includes('report')) badgeClass += ' report';

             record.innerHTML = `
                 <div class="document-record-info">
                     <i class="fas ${getFileIcon(doc)} document-record-icon"></i>
                     <div class="document-record-details">
                         <div class="document-record-name">${doc.name || 'Sin Nombre'}</div>
                         <div class="document-record-meta">Subido el ${doc.uploadDate ? new Date(doc.uploadDate).toLocaleDateString('es-ES', { day: 'numeric', month: 'short' }) : ''} • ${formatFileSize(doc.size || 0)}</div>
                     </div>
                 </div>
                 <div class="document-record-actions">
                     <span class="${badgeClass}">${doc.type || 'Archivo'}</span>
                     ${doc.analysis ? '<i class="fas fa-check-circle" style="color: var(--success);" title="Analizado"></i>' : ''}
                 </div>
             `;
             // Add click listener to view/analyze the document
             record.addEventListener('click', () => {
                 // Logic to switch to document tab and display this document/analysis
                 console.log(`Clicked document: ${doc.id}`);
                 // Example: Switch tab and show preview
                 switchTab('documents');
                 displayDocumentPreview(doc); // You'd need this function
             });
             docRecordsContainer.appendChild(record);
         });
     }

     // Function to display a specific document in the preview area
     function displayDocumentPreview(doc) {
          if (!docPreview || !docNameEl || !docContentEl || !docAnalysisEl || !analysisContentEl) return;

          currentFileId = doc.id; // Set the current file context

          docNameEl.textContent = doc.name || 'Documento';
          // Show a placeholder in the content area
          docContentEl.innerHTML = `
              <div style="padding: 2rem; text-align: center; color: var(--text-medium);">
                  <i class="fas ${getFileIcon(doc)} fa-3x" style="color: var(--primary-color); margin-bottom: 1rem;"></i>
                  <p>Previsualización de: ${doc.name}</p>
                  <p style="font-size: 0.8rem;">(Previsualización real no implementada en esta demo)</p>
              </div>`;

          // Show analysis if available
          if (doc.analysis) {
              displayDocumentAnalysis(doc.analysis);
              docAnalysisEl.classList.add('active');
          } else {
              analysisContentEl.innerHTML = '<p style="color: var(--text-light);">Este documento aún no ha sido analizado.</p>';
              docAnalysisEl.classList.remove('active'); // Hide analysis section
              currentAnalysis = null;
          }

          docPreview.classList.add('active'); // Show the preview section
      }

      // Function to display analysis results
      function displayDocumentAnalysis(analysis) {
          if (!analysisContentEl) return;
          // Build HTML for analysis (similar to previous implementations)
          let html = `<p><strong>Tipo:</strong> ${analysis.type || 'Desconocido'}</p>`;
          // ... add sections for results, recommendations etc. ...
          analysisContentEl.innerHTML = html;
          currentAnalysis = analysis; // Store for context
      }

     // Function to switch tabs programmatically
     function switchTab(tabDataValue) {
         const tabToActivate = document.querySelector(`.assistant-tab[data-tab="${tabDataValue}"]`);
         if (tabToActivate) {
             tabToActivate.click(); // Simulate click to trigger standard tab switching
         }
     }


    // Assistant Tab Switching Logic
     assistantTabs.forEach(tab => {
         tab.addEventListener('click', () => {
             assistantTabs.forEach(t => t.classList.remove('active'));
             tabContents.forEach(content => { if(content) content.style.display = 'none'; });
             tab.classList.add('active');
             const targetTabId = tab.getAttribute('data-tab') + '-tab';
             const targetContent = document.getElementById(targetTabId);
             if (targetContent) { targetContent.style.display = 'block'; }
             else { console.warn(`Tab content not found for ID: ${targetTabId}`); }
         });
     });
     // Ensure initial tab display
     const initialActiveAssistantTab=document.querySelector('.assistant-tab.active');
     if(initialActiveAssistantTab){
          const initialTargetId=initialActiveAssistantTab.getAttribute('data-tab')+'-tab';
          const initialTargetContent=document.getElementById(initialTargetId);
          if(initialTargetContent)initialTargetContent.style.display='block';
     } else if(assistantTabs.length > 0){
          assistantTabs[0].click();
     }

    // Chat Logic Handlers (addChatMessage, showTyping, hideTyping, handleSendMessage)
    const addChatMessage = (content, sender) => { /* ... as defined in main.js or helpers.js ... */
         if (!chatContainer) return; const messageDiv = document.createElement('div'); messageDiv.className = `chat-message ${sender}`; const bubble = document.createElement('div'); bubble.className = 'message-bubble'; bubble.innerHTML = typeof content === 'string' && content.match(/<[a-z][\s\S]*>/i) ? content : `<p>${content || ''}</p>`; messageDiv.appendChild(bubble); chatContainer.appendChild(messageDiv); chatContainer.scrollTop = chatContainer.scrollHeight;
     };
    const showTyping = () => { /* ... as defined in main.js or helpers.js ... */
         if (!chatContainer || chatContainer.querySelector('.typing-indicator')) return; const typingIndicator = document.createElement('div'); typingIndicator.className = 'typing-indicator'; typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>'; chatContainer.appendChild(typingIndicator); chatContainer.scrollTop = chatContainer.scrollHeight;
     };
    const hideTyping = () => { chatContainer?.querySelector('.typing-indicator')?.remove(); };

    const handleSendMessage = async () => { /* ... as defined before ... */
         if (!chatInput || !chatInput.value.trim()) return; const message = chatInput.value.trim(); addChatMessage(message, 'user'); chatInput.value = ''; chatInput.style.height = 'auto'; showTyping(); try { await new Promise(resolve => setTimeout(resolve, 1500)); const aiResponse = `Respuesta simulada para: "${message.substring(0,30)}...".`; hideTyping(); addChatMessage(aiResponse, 'ai'); } catch (error) { hideTyping(); showToast('Error Chat', 'No se pudo obtener respuesta.', 'error'); console.error("Chat error:", error); }
     };
    if (sendButton) sendButton.addEventListener('click', handleSendMessage);
    if (chatInput) {
        chatInput.addEventListener('keypress', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } });
        chatInput.addEventListener('input', () => { chatInput.style.height = 'auto'; chatInput.style.height = (chatInput.scrollHeight) + 'px'; });
    }
    if (voiceButton) voiceButton.addEventListener('click', () => showToast('Info', 'Entrada de voz no implementada.', 'info'));
    if (attachButton) attachButton.addEventListener('click', () => fileInput?.click());
    if (fileInput) fileInput.addEventListener('change', (event) => { /* Handle file attachment for chat */ });

    // Document Tab Logic Handlers
    if(uploadArea){/* ... Add drag/drop listeners ... */}
    if(browseFilesBtn) browseFilesBtn.addEventListener('click', () => docFileInput?.click());
    if(docFileInput) docFileInput.addEventListener('change', (event) => { /* Handle file upload for document tab */ });
    if(analyzeDocBtn) analyzeDocBtn.addEventListener('click', () => { /* Logic to analyze currentFileId */ showToast('Info', 'Análisis no implementado.', 'info'); });
    if(integrateBtn) integrateBtn.addEventListener('click', () => { /* Logic to integrate currentAnalysis */ showToast('Info', 'Integración no implementada.', 'info'); });


    // --- Initializations ---
    loadPatientData(); // Load data when dashboard JS runs

    console.log("Dashboard specific logic initialized.");

}); // End Dashboard specific DOMContentLoaded