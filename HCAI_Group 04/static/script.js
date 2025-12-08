document.addEventListener("DOMContentLoaded", function () {

    const uploadBtn = document.getElementById("Button");
    const takePhotoBtn = document.getElementById("Button_0");
    const infoText = document.getElementById("infoText");
    const fileInput = document.getElementById("fileInput");
    const logoImg = document.getElementById("logo");
    const controls = document.getElementById("Container_4");
    const heading = document.getElementById("Heading_2");
    const subText = document.getElementById("Container_3");
    const supportedText = document.getElementById("Container_5");

    const previewWrapper = document.createElement("div");
    previewWrapper.style.position = "relative";
    previewWrapper.style.width = "100%";
    previewWrapper.style.height = "100%";
    previewWrapper.style.display = "flex";
    logoImg.parentNode.insertBefore(previewWrapper, logoImg);
    previewWrapper.appendChild(logoImg);

    function cloneButtonStyle(targetBtn, sourceBtn) {
        const s = window.getComputedStyle(sourceBtn);
        Object.assign(targetBtn.style, {
            width: s.width,
            height: s.height,
            borderRadius: s.borderRadius,
            background: s.background,
            color: s.color,
            fontSize: s.fontSize,
            fontWeight: s.fontWeight,
            border: s.border,
            padding: s.padding,
            cursor: "pointer"
        });
    }

    const btnContainer = document.createElement("div");
    btnContainer.style.display = "none";
    btnContainer.style.flexDirection = "row";
    btnContainer.style.justifyContent = "center";
    btnContainer.style.alignItems = "center";
    btnContainer.style.width = "100%";
    btnContainer.style.marginTop = "1rem";
    btnContainer.style.gap = "1rem";
    controls.appendChild(btnContainer);

    const detailsBtn = document.createElement("button");
    detailsBtn.innerHTML = `<span class="text">View Details</span>`;
    cloneButtonStyle(detailsBtn, uploadBtn);
    btnContainer.appendChild(detailsBtn);

    const newImageBtn = document.createElement("button");
    newImageBtn.innerHTML = `<span class="text">Choose Image</span>`;
    cloneButtonStyle(newImageBtn, uploadBtn);
    btnContainer.appendChild(newImageBtn);

    const deleteBtn = document.createElement("div");
    deleteBtn.innerHTML = "✖";
    Object.assign(deleteBtn.style, {
        position: "absolute",
        top: "8px",
        right: "8px",
        width: "32px",
        height: "32px",
        display: "none",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "18px",
        fontWeight: "bold",
        color: "white",
        background: "rgba(0,0,0,0.7)",
        borderRadius: "50%",
        cursor: "pointer"
    });
    previewWrapper.appendChild(deleteBtn);

    uploadBtn.addEventListener("click", () => fileInput.click());
    newImageBtn.addEventListener("click", () => fileInput.click());

    function validateImage(file) {
        const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (!allowedTypes.includes(file.type)) return false;
        if (file.size > maxSize) return false;
        return true;
    }

    fileInput.addEventListener("change", async function () {
        const file = this.files[0];
        if (!file) return;
        infoText.textContent = "";

        if (!validateImage(file)) {
            infoText.style.color = "red";
            infoText.textContent = "❌ Only JPG, PNG, WebP and max 10MB allowed.";
            fileInput.value = "";
            return;
        }

        // Preview immediately
        const reader = new FileReader();
        reader.onload = function (e) {
            logoImg.src = e.target.result;
            logoImg.style.objectFit = "cover";
            logoImg.style.width = "100%";
            logoImg.style.height = "100%";
            logoImg.style.borderRadius = "1rem";

            uploadBtn.style.display = "none";
            takePhotoBtn.style.display = "none";
            heading.style.display = "none";
            subText.style.display = "none";
            supportedText.style.display = "none";
            btnContainer.style.display = "flex";
            deleteBtn.style.display = "flex";
        };
        reader.readAsDataURL(file);

        // Send to backend for animal validation
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                infoText.style.color = "red";
                infoText.textContent = data.error;
                logoImg.src = "static/uploaded_images/logo.png";
                logoImg.style.objectFit = "contain";
                deleteBtn.style.display = "none";
                btnContainer.style.display = "none";
                uploadBtn.style.display = "block";
                takePhotoBtn.style.display = "block";
                heading.style.display = "block";
                subText.style.display = "block";
                supportedText.style.display = "block";
                fileInput.value = "";
                return;
            }

            // Save prediction for details page
            localStorage.setItem("uploadedImage", logoImg.src);
            localStorage.setItem("prediction", data.prediction);

        } catch (err) {
            infoText.style.color = "red";
            infoText.textContent = "❌ Server error. Please try again.";
        }
    });

    detailsBtn.addEventListener("click", () => {
        if (!logoImg.src) return;
        window.location.href = "/prediction";
    });

    deleteBtn.addEventListener("click", () => {
        logoImg.src = "static/uploaded_images/logo.png";
        logoImg.style.objectFit = "contain";
        deleteBtn.style.display = "none";
        btnContainer.style.display = "none";
        uploadBtn.style.display = "block";
        takePhotoBtn.style.display = "block";
        heading.style.display = "block";
        subText.style.display = "block";
        supportedText.style.display = "block";
        fileInput.value = "";
        infoText.textContent = "";
    });

});
