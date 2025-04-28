document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('YOUR_OCR_API_ENDPOINT', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            displayResults(data.text);
        } catch (error) {
            console.error('Error processing the file:', error);
        }
    } else {
        alert('Por favor, selecciona un archivo.');
    }
});

function displayResults(text) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `<pre>${text}</pre>`;
}
