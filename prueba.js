document.addEventListener('DOMContentLoaded', function() {
    const generarPlanBtn = document.getElementById('generarPlan');
    const planContenidoDiv = document.getElementById('plan-contenido');

    generarPlanBtn.addEventListener('click', function() {
        const edad = document.getElementById('edad').value;
        const peso = document.getElementById('peso').value;
        const altura = document.getElementById('altura').value;
        const condiciones = document.getElementById('condiciones').value;
        const resultados = document.getElementById('resultados').value;

        // Aquí simularíamos la llamada a la IA para generar el plan
        const planGenerado = `
            <h3>Plan de Tratamiento y Nutrición Personalizado</h3>
            <p><strong>Edad:</strong> ${edad} años</p>
            <p><strong>Peso:</strong> ${peso} kg</p>
            <p><strong>Altura:</strong> ${altura} cm</p>
            <p><strong>Condiciones Médicas:</strong> ${condiciones}</p>
            <p><strong>Resultados de Análisis:</strong> ${resultados}</p>
            <p><strong>Recomendaciones de Tratamiento:</strong> (Esto lo generaría la IA)</p>
            <p><strong>Recomendaciones Nutricionales:</strong> (Esto también lo generaría la IA)</p>
        `;

        planContenidoDiv.innerHTML = planGenerado;
    });
});