// Detección de modo oscuro según preferencias del sistema
if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.classList.add('dark');
  }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
    if (event.matches) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  });
  
  // Alternar modo oscuro al hacer clic en el ícono de la luna
  const themeToggle = document.getElementById('theme-toggle');
  if(themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.documentElement.classList.toggle('dark');
    });
  }
  
  // Función para colapsar/expandir el contenido de cada card
  function toggleCardBody(header) {
    header.classList.toggle('active');
    const cardBody = header.nextElementSibling;
    cardBody.classList.toggle('hidden');
    // Actualizar atributo ARIA para accesibilidad
    header.setAttribute('aria-expanded', header.classList.contains('active'));
  }
  
  // Manejo del envío del formulario (demo)
  const form = document.getElementById('form-guia-paciente');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      alert('Guía del Paciente guardada (demostración).');
      // Lógica real para guardar o enviar datos iría aquí.
    });
  }
  
  // Funciones para el modal de historias
  function closeModal() {
    document.getElementById('historias-modal').style.display = 'none';
  }
  
  function saveNotes() {
    alert('Notas guardadas');
  }
  
  // Inicialización de gráfico con Chart.js para Tendencias Generales
  document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('trendChart');
    if (ctx) {
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May'],
          datasets: [
            {
              label: 'Edad Biológica',
              data: [65, 66, 64, 67, 66],
              borderColor: 'rgba(41, 59, 100, 0.9)',
              fill: false
            },
            {
              label: 'Medición de Telómeros',
              data: [50, 52, 51, 53, 52],
              borderColor: 'rgba(75, 192, 192, 0.9)',
              fill: false
            },
            {
              label: 'Edad Genética',
              data: [60, 61, 59, 62, 61],
              borderColor: 'rgba(255, 159, 64, 0.9)',
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: false
            }
          }
        }
      });
    }
  });
  