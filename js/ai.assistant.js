// ai_assistant.js

// Simulación de respuestas de IA (puedes reemplazar esta lógica con llamadas reales a una API de IA)
const aiResponses = [
    "He analizado los datos y parece que el paciente presenta un riesgo moderado de enfermedad cardiovascular.",
    "Los resultados de laboratorio muestran niveles óptimos en la mayoría de los biomarcadores, pero es recomendable monitorear la presión arterial.",
    "La tendencia indica una ligera diferencia entre la edad cronológica y biológica; se sugiere evaluar hábitos de vida.",
    "Recomiendo revisar la medicación y realizar un seguimiento en 3 meses para confirmar los resultados.",
    "Según el análisis, se observa un incremento en la inflamación sistémica, lo que podría requerir una intervención temprana."
  ];
  
  function getAIResponse() {
    const randomIndex = Math.floor(Math.random() * aiResponses.length);
    const response = aiResponses[randomIndex];
    console.log("AI Response generated: ", response);
    return response;
  }
  
  document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
  
    // Función para agregar un mensaje al chat
    function addMessage(text, sender) {
      const messageDiv = document.createElement("div");
      messageDiv.classList.add("message", sender);
      messageDiv.innerHTML = text;
      chatWindow.appendChild(messageDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  
    // Escuchar el envío del formulario
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const userText = userInput.value.trim();
      console.log("User input: ", userText);
      if (!userText) return;
      addMessage(userText, "user");
      userInput.value = "";
      // Simula un retardo para la respuesta de la IA
      setTimeout(() => {
        const aiText = getAIResponse();
        if (aiText && aiText.trim() !== "") {
          addMessage(aiText, "ai");
        } else {
          addMessage("Lo siento, no pude procesar la respuesta.", "ai");
        }
      }, 1000);
    });
  });
  