const interval = 3000;
const timer = document.getElementById('timer');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

const resultBox = document.getElementById('result-box');
const confidenceBox = document.getElementById('confidence-box');
const pauseButton = document.getElementById('pause-button');
const continuarButton = document.getElementById('continuar');
const anuncio = document.getElementById('anuncio');
const container = document.getElementById('daWholeThing');
let isProcessing = false;
let isPaused = true;
        
continuarButton.addEventListener('click', (event) => {
    event.preventDefault();
    isPaused = !isPaused;
    anuncio.style.display = 'none';
    container.style.display = 'inline-block';
});

function clicked(toPause) {
    if(toPause) {
        isPaused = !isPaused;
        pauseButton.textContent = isPaused ? "▶️" : "⏸️";
    } else {
        resultBox.textContent = '';
        confidenceBox.textContent = '';
    }
}

navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } }).then(stream => { 
    video.srcObject = stream; 
    resultBox.innerText = ""; 
    let innterInterval = interval;
    setInterval(() => {
    // 1. Si está pausado, detenemos el reloj y NO mandamos fotos
    if (isPaused) {
        timer.innerHTML = 'Procesamiento en pausa';
        return; // <-- Bloquea todo lo que está abajo
    }

    // 2. Si no está pausado, corre el flujo normal
    if (innterInterval >= 1000) {
        timer.innerHTML = `La próxima captura será tomada dentro de ${innterInterval/1000} segundos`;
        innterInterval -= 1000;
    } else {
        timer.innerHTML = 'Prediciendo...';
        captureAndPredict(); 
        innterInterval = interval;
        }
    }, 1000);
})
    .catch(err => { 
        console.error("Error de cámara:", err); 
        resultBox.innerText = "No se pudo acceder a la cámara"; 
    });

        function captureAndPredict() {
            if (isProcessing) return;

            isProcessing = true;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');
                fetch('https://wasteit.onrender.com/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if(data.error) {
                        resultBox.innerText = "Error: " + data.error;
                    } else {
                        // Mostramos el resultado actualizado
                        resultBox.innerHTML = `Detected as <span class="${data.class}">${data.class.toUpperCase()}</span> <br> <small>Confianza: ${data.confidence}</small>`;
                    }
                })
                .catch(err => {
                    console.error("Error:", err);
                    resultBox.innerText = "Server error";
                })
                .finally(() => {
                    // Available
                    isProcessing = false;
                });
            }, 'image/jpeg', 0.6);
        }