document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const modelSelector = document.getElementById("model-selector");
    const resultDiv = document.getElementById("result");

    // Helper function to handle image preview
    function displayImagePreview(image) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.src = e.target.result;
            img.width = 300;
            img.height = 300;
            dropZone.innerHTML = "";
            dropZone.appendChild(img);
        };
        reader.readAsDataURL(image);
    }

    // Helper function to handle file upload via input
    fileInput.addEventListener("change", function (e) {
        const file = e.target.files[0];
        if (file) {
            displayImagePreview(file);
            classifyImage(file);
        }
    });

    // Function to handle drag and drop events
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();

        const dataTransfer = e.dataTransfer;
        const files = dataTransfer.files;

        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith("image/")) {
                displayImagePreview(file);
                classifyImage(file);
            }
        }
    }

    // Drag and drop event listeners
    dropZone.addEventListener("dragenter", (e) => e.preventDefault());
    dropZone.addEventListener("dragover", (e) => e.preventDefault());
    dropZone.addEventListener("drop", handleDrop);

    // Click event listener to trigger file input
    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    async function classifyImage(image) {
        const model = modelSelector.value;
        const reader = new FileReader();

        reader.onload = async function (e) {
            const imageDataUrl = e.target.result;
            const base64Image = imageDataUrl.split(",")[1];
            const prediction = await eel.classify_image(base64Image, model)();
            resultDiv.innerHTML = `<p>Model: ${model}</p><p>Result: ${prediction}</p>`;
        };

        reader.readAsDataURL(image);
    }
});
