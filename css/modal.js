<!-- Modal para Asistente IA -->
<div id="ai-assistant-modal" class="modal hidden">
    <div class="modal-content">
        <span class="close-button" onclick="closeAIAssistant()">&times;</span>
        <h2>Asistente IA</h2>
        <form id="file-upload-form">
            <input type="file" id="file-input" multiple accept=".pdf,.txt,.docx">
            <button type="button" onclick="uploadFiles()">Analizar Archivos</button>
        </form>
        <div id="analysis-results"></div>
    </div>
</div>

<script>
function openAIAssistant() {
    document.getElementById('ai-assistant-modal').classList.remove('hidden');
}

function closeAIAssistant() {
    document.getElementById('ai-assistant-modal').classList.add('hidden');
}

async function uploadFiles() {
    const fileInput = document.getElementById('file-input');
    const files = fileInput.files;
    const resultsDiv = document.getElementById('analysis-results');
    resultsDiv.innerHTML = ''; // Limpiar resultados anteriores

    for (const file of files) {
        const reader = new FileReader();
        reader.onload = async function(e) {
            const content = e.target.result;
            const analysis = await analyzeWithAI(content, file.name);
            displayAnalysis(analysis, file.name);
        };
        reader.readAsText(file);
    }
}

async function analyzeWithAI(content, fileName) {
    // Aquí se debe integrar con una API de IA avanzada como Gemini Pro
    // Por ahora, simulamos un análisis
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(`Análisis del archivo ${fileName}: ${content.substring(0, 100)}...`);
        }, 1000);
    });
}

function displayAnalysis(analysis, fileName) {
    const resultsDiv = document.getElementById('analysis-results');
    const resultItem = document.createElement('div');
    resultItem.innerHTML = `<strong>${fileName}</strong><p>${analysis}</p>`;
    resultsDiv.appendChild(resultItem);
}
</script>
