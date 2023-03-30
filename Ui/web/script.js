document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const modelSelector = document.getElementById("model-selector");
    const classifyBtn = document.getElementById("classify-btn");
    const removeBtn = document.getElementById("remove-btn");
    const resultDiv = document.getElementById("result");
    const imageNameDiv = document.getElementById("image-name");
    let uploadedImage = null;

function displayImagePreview(image) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.src = e.target.result;

        img.onload = function () {
            const maxHeight = 300;
            const aspectRatio = img.width / img.height;

            if (img.height > maxHeight) {
                img.height = maxHeight;
                img.width = maxHeight * aspectRatio;
            }

            const imageNameDiv = document.getElementById("image-name");
            imageNameDiv.innerText = image.name;
            imageNameDiv.style.fontWeight = "bold";

            dropZone.innerHTML = "";
            dropZone.appendChild(img);
        };
    };
    reader.readAsDataURL(image);
}


    fileInput.addEventListener("change", function (e) {
        const file = e.target.files[0];
        if (file) {
            uploadedImage = file;
            displayImagePreview(file);
        }
    });

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();

        const dataTransfer = e.dataTransfer;
        const files = dataTransfer.files;

        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith("image/")) {
                uploadedImage = file;
                displayImagePreview(file);
            }
        }
    }

    dropZone.addEventListener("dragenter", (e) => e.preventDefault());
    dropZone.addEventListener("dragover", (e) => e.preventDefault());
    dropZone.addEventListener("drop", handleDrop);

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    classifyBtn.addEventListener("click", async () => {
        if (uploadedImage) {
            classifyImage(uploadedImage);
        } else {
            alert("Please upload an image first");
        }
    });

    removeBtn.addEventListener("click", () => {
    uploadedImage = null;
    fileInput.value = ""; // Add this line to reset the file input value
    dropZone.innerHTML = "<p>Drag and drop your image here, or click to select a file</p>";
    });

    async function classifyImage(image) {
        const model = modelSelector.value;
        const reader = new FileReader();

        reader.onload = async function (e) {
            const imageDataUrl = e.target.result;
            const base64Image = imageDataUrl.split(",")[1];
            const prediction = await eel.classify_image(base64Image, model)();
            displayResult(prediction);
        };

        reader.readAsDataURL(image);
    }

    function displayResult(prediction) {
        let resultHtml = "";
        for (const [className, percentage] of prediction) {
            resultHtml += `<p><strong>${className}</strong>: ${percentage.toFixed(2)}%</p>`;
        }
        resultDiv.innerHTML = resultHtml;
    }
});
