// ============ Datos simulados ============ //
const patientsData = [
    // Cada objeto simula un paciente con datos de Edad Cronológica, Edad Biológica y nivel de Telómeros
    { name: "John", chronoAge: 35, bioAge: 32, telomeres: 51 },
    { name: "Alice", chronoAge: 40, bioAge: 38, telomeres: 55 },
    { name: "Bob", chronoAge: 50, bioAge: 60, telomeres: 47 },
    { name: "Diana", chronoAge: 45, bioAge: 44, telomeres: 53 },
    { name: "Carlos", chronoAge: 60, bioAge: 58, telomeres: 50 },
    { name: "Elena", chronoAge: 65, bioAge: 62, telomeres: 48 },
    { name: "Frank", chronoAge: 70, bioAge: 68, telomeres: 46 },
    { name: "Grace", chronoAge: 55, bioAge: 50, telomeres: 57 },
    // ... puedes añadir más datos o traerlos de tu backend
  ];
  
  // ============ Función para renderizar Scatter 3D ============ //
  function render3DScatter(dataArray) {
    const trace = {
      x: dataArray.map(d => d.chronoAge),
      y: dataArray.map(d => d.bioAge),
      z: dataArray.map(d => d.telomeres),
      mode: 'markers',
      type: 'scatter3d',
      text: dataArray.map(d => d.name), // nombre del paciente
      marker: {
        size: 5,
        color: dataArray.map(d => d.bioAge), // colorear según la edad biológica
        colorscale: 'Portland',
        opacity: 0.8
      }
    };
  
    const layout = {
      title: 'Edad Cronológica vs. Edad Biológica vs. Telómeros',
      scene: {
        xaxis: { title: 'Edad Cronológica (años)' },
        yaxis: { title: 'Edad Biológica (años)' },
        zaxis: { title: 'Nivel de Telómeros' }
      },
      margin: { l: 0, r: 0, b: 0, t: 40 }
    };
  
    Plotly.newPlot('scatter3d', [trace], layout);
  }
  
  // ============ Función para renderizar Barras ============ //
  function renderBarChart(dataArray) {
    // Agrupemos datos por "nombre" para ver una comparación de bioAge vs. telomeres
    // O puedes graficar cronAge, bioAge, telomeres en un bar grouping
  
    const names = dataArray.map(d => d.name);
    const bioAges = dataArray.map(d => d.bioAge);
    const telomeres = dataArray.map(d => d.telomeres);
  
    const traceBioAge = {
      x: names,
      y: bioAges,
      name: 'Edad Biológica',
      type: 'bar'
    };
  
    const traceTelomeres = {
      x: names,
      y: telomeres,
      name: 'Telómeros',
      type: 'bar'
    };
  
    const data = [traceBioAge, traceTelomeres];
  
    const layout = {
      barmode: 'group',
      title: 'Comparación de Edad Biológica y Telómeros',
      xaxis: { title: 'Pacientes' },
      yaxis: { title: 'Valores' }
    };
  
    Plotly.newPlot('barChart', data, layout);
  }
  
  // ============ Filtrado por nombre de paciente ============ //
  function filterByName(name) {
    if (!name) return patientsData; // si no hay texto, devolver todos
    return patientsData.filter(p => p.name.toLowerCase().includes(name.toLowerCase()));
  }
  
  // ============ Inicializar al cargar ============ //
  document.addEventListener('DOMContentLoaded', () => {
    // Renderizar gráficos con TODOS los datos inicialmente
    render3DScatter(patientsData);
    renderBarChart(patientsData);
  
    // Agregar evento al botón de búsqueda
    const btnSearch = document.getElementById('btnSearch');
    const searchInput = document.getElementById('patientSearch');
  
    btnSearch.addEventListener('click', () => {
      const query = searchInput.value.trim();
      const filtered = filterByName(query);
      // Volver a renderizar con los datos filtrados
      render3DScatter(filtered);
      renderBarChart(filtered);
    });
  });
  