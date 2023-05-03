document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const modelSelector = document.getElementById("model-selector");
    const classifyBtn = document.getElementById("classify-btn");
    const removeBtn = document.getElementById("remove-btn");
    const resultDiv = document.getElementById("result-content");
    const resultMessage = document.getElementById("result-message");
    const darkModeBtn = document.getElementById("dark-mode-btn");
    const darkModeStylesheet = document.getElementById("dark-mode-stylesheet");
    const imageNameDiv = document.getElementById("image-name");
    let uploadedImage = null;

if (localStorage.getItem("darkMode") === "enabled") {
  enableDarkMode();
}

// Toggle dark mode on button click
darkModeBtn.addEventListener("click", () => {
  if (localStorage.getItem("darkMode") === "enabled") {
    disableDarkMode();
  } else {
    enableDarkMode();
  }
});

function enableDarkMode() {
  darkModeStylesheet.disabled = false;
  document.body.classList.add("dark-mode");
  document.querySelector(".container").classList.add("dark-mode");
  document.querySelector(".drop-zone").classList.add("dark-mode");
  document.querySelector(".result").classList.add("dark-mode");
  document.getElementById("model-selector").classList.add("dark-mode");
  document.getElementById("dark-mode-btn").classList.add("dark-mode");
  document.querySelector("h1").classList.add("dark-mode");
  localStorage.setItem("darkMode", "enabled");
}

function disableDarkMode() {
  darkModeStylesheet.disabled = true;
  document.body.classList.remove("dark-mode");
  document.querySelector(".container").classList.remove("dark-mode");
  document.querySelector(".drop-zone").classList.remove("dark-mode");
  document.querySelector(".result").classList.remove("dark-mode");
  document.getElementById("model-selector").classList.remove("dark-mode");
  document.getElementById("dark-mode-btn").classList.remove("dark-mode");
  document.querySelector("h1").classList.remove("dark-mode");
  localStorage.setItem("darkMode", "disabled");
}

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
    fileInput.value = "";
    dropZone.innerHTML = "<p>Drag and drop your image here, or click to select a file</p>";
    resultMessage.style.display = "block";
    resultDiv.style.display = "none";
    });

    async function classifyImage(image) {
    const model = modelSelector.value;
    const reader = new FileReader();

    // Show loading spinner
    document.getElementById("loading-spinner").style.display = "flex";

    reader.onload = async function (e) {
        const imageDataUrl = e.target.result;
        const base64Image = imageDataUrl.split(",")[1];
        const prediction = await eel.classify_image(base64Image, model)();
        resultDiv.innerHTML = "";
          resultMessage.style.display = "none";
        resultDiv.style.display = "block";
        displayResult(prediction);

        // Hide loading spinner
        document.getElementById("loading-spinner").style.display = "none";
    };

    reader.readAsDataURL(image);
}

    function displayResult(prediction) {
    let resultHtml = "";
    for (const [className, percentage] of prediction) {
        resultHtml += `
            <div class="percentage-bar-container">
                <strong class="class-name">${className}</strong>
                <div class="percentage-bar" data-percentage="${percentage}">
                    <div class="percentage-bar-inner" style="width: ${percentage}%;"></div>
                </div>
                <span class="percentage-value">${percentage.toFixed(2)}%</span>
            </div>
        `;
    }
    resultDiv.innerHTML = resultHtml;
}

});
function showLoading() {
    const loadingDiv = document.getElementById("loading");
    loadingDiv.style.display = "block";
}

function hideLoading() {
    const loadingDiv = document.getElementById("loading");
    loadingDiv.style.display = "none";
}
eel.expose(showLoading);
eel.expose(hideLoading);
