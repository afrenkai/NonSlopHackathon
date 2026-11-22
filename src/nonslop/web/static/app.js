document.addEventListener("DOMContentLoaded", () => {
  const yearInput = document.getElementById("year");
  if (yearInput) {
    yearInput.max = String(new Date().getFullYear());
    yearInput.placeholder = String(new Date().getFullYear() - 20);
  }
  const referencePickers = [...document.querySelectorAll("[data-reference-picker]")];
  function showReferences() {
    const opening = document.getElementById("reference-image");
    const count = (opening?.files.length || 0) + referencePickers.reduce((total, picker) => total + picker.files.length, 0);
    for (const picker of referencePickers) {
      picker.setCustomValidity(count > 8 ? "Choose at most 8 reference images total." : "");
    }
    const selection = document.getElementById("reference-selection");
    const ignored = referencePickers.reduce((total, picker) => total + Number(picker.dataset.ignoredFiles || 0), 0);
    if (selection) selection.textContent = `${count} reference ${count === 1 ? "image" : "images"} selected.${ignored ? ` ${ignored} unsupported or hidden files skipped.` : ""}`;
  }
  for (const picker of referencePickers) {
    picker.addEventListener("change", () => {
      if (picker.hasAttribute("webkitdirectory")) {
        const images = new DataTransfer();
        for (const file of picker.files) {
          if (["image/png", "image/jpeg", "image/webp"].includes(file.type) || /\.(png|jpe?g|jfif|webp)$/i.test(file.name) || !file.name.includes(".")) images.items.add(file);
        }
        picker.dataset.ignoredFiles = String(picker.files.length - images.files.length);
        picker.files = images.files;
      }
      showReferences();
    });
  }
  document.getElementById("reference-image")?.addEventListener("change", showReferences);
});

document.addEventListener("htmx:beforeRequest", (event) => {
  event.detail.elt?.closest("form")?.querySelector("[data-form-feedback]")?.remove();
});

document.addEventListener("htmx:beforeSwap", (event) => {
  const response = event.detail.xhr;
  if ([400, 403, 409, 413, 422].includes(response.status)) {
    // Keep the archive (and its swap target) intact when a form needs a correction.
    const form = event.detail.requestConfig?.elt?.closest("form");
    if (form) {
      let feedback = form.querySelector("[data-form-feedback]");
      if (!feedback) {
        feedback = document.createElement("div");
        feedback.dataset.formFeedback = "";
        form.append(feedback);
      }
      event.detail.target = feedback;
      event.detail.swapOverride = "innerHTML";
    }
    event.detail.shouldSwap = true;
    event.detail.isError = false;
  }
});

