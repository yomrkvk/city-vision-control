// Пока база
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const resultContainer = document.getElementById('resultContainer');

uploadBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) {
        alert('Пожалуйста, выберите изображение');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    resultContainer.innerHTML = '<p>Обработка...</p>';

    try {
        const response = await fetch('http://localhost:8000/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Ошибка: ${response.status}`);

        const data = await response.json();

        // Вывод JSON красиво
        resultContainer.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
    } catch (err) {
        resultContainer.innerHTML = `<p style="color: red;">${err.message}</p>`;
    }
});
