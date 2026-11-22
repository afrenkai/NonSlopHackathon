// A bookmark can target a new document or a fragment on an already-open landing page.
function followMovedSection() {
  if (location.pathname !== "/") return;
  if (location.hash === "#your-box") location.replace("/dashboard#box");
  else if (location.hash === "#new-visit") location.replace("/visits/new");
}
window.addEventListener("hashchange", followMovedSection);
document.addEventListener("DOMContentLoaded", () => {
  followMovedSection();
  for (const opener of document.querySelectorAll('button[aria-haspopup="dialog"]')) {
    const dialog = document.getElementById(opener.getAttribute('aria-controls'));
    if (!(dialog instanceof HTMLDialogElement)) continue;
    opener.addEventListener('click', () => dialog.showModal());
    dialog.querySelector('[data-close-dialog]').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => opener.focus());
    const url = new URL(location.href);
    if (dialog.id === `add-${url.searchParams.get('add')}`) {
      dialog.showModal();
      url.searchParams.delete('add');
      history.replaceState(history.state, '', url);
    }
  }
  const yearInput = document.getElementById("year");
  if (yearInput) {
    yearInput.max = String(new Date().getFullYear());
    yearInput.placeholder = String(new Date().getFullYear() - 20);
  }
  const opening = document.getElementById("reference-image");
  if (!opening) return;

  const locationInput = document.getElementById("location");
  const descriptionInput = document.getElementById("description");
  const useWeb = document.getElementById("use-web");
  const webUnavailable = useWeb.disabled;
  const preview = document.getElementById("reference-preview");
  const presets = [...document.querySelectorAll('input[name="visit_preset"]')];
  const referencePickers = [...document.querySelectorAll("[data-reference-picker]")];
  let activePreset;
  let customFields;
  let previewObjectUrl;

  function showReferences() {
    const files = referencePickers.flatMap((picker) => [...picker.files]);
    const count = activePreset ? 1 : opening.files.length + files.length;
    descriptionInput.required = count === 0;
    for (const picker of [opening, ...referencePickers]) {
      picker.setCustomValidity(count > 8 ? "Choose at most 8 reference images total." : "");
    }
    const selection = document.getElementById("reference-selection");
    const ignored = referencePickers.reduce((total, picker) => total + Number(picker.dataset.ignoredFiles || 0), 0);
    selection.textContent = `${count} reference ${count === 1 ? "image" : "images"} selected.${ignored ? ` ${ignored} unsupported or hidden files skipped.` : ""}`;

    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = undefined;
    if (!activePreset) {
      const firstAdditional = files.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "accent" }))[0];
      const file = opening.files[0] ?? firstAdditional;
      if (file) previewObjectUrl = URL.createObjectURL(file);
    }
    const photo = activePreset?.dataset.photo ?? previewObjectUrl;
    preview.hidden = !photo;
    preview.alt = activePreset?.dataset.location ?? "Uploaded photograph preview";
    if (photo) preview.src = photo;
    else preview.removeAttribute("src");
  }

  function selectVisit() {
    const preset = presets.find((option) => option.checked && option.value);
    if (preset && !activePreset) {
      customFields = { location: locationInput.value, year: yearInput.value, useWeb: useWeb.checked };
    }
    if (preset || activePreset) {
      locationInput.value = preset?.dataset.location ?? customFields.location;
      yearInput.value = preset?.dataset.year ?? customFields.year;
      useWeb.checked = preset ? false : customFields.useWeb;
    }
    activePreset = preset;
    const isDemo = Boolean(preset);
    for (const input of [locationInput, yearInput]) {
      input.readOnly = isDemo;
      input.required = !isDemo;
    }
    for (const picker of [opening, ...referencePickers]) picker.disabled = isDemo;
    useWeb.disabled = isDemo || webUnavailable;
    document.getElementById("reference-upload").hidden = isDemo;
    document.getElementById("reference-options").hidden = isDemo;
    document.getElementById("web-context-help").hidden = isDemo;
    document.getElementById("demo-context-help").hidden = !isDemo;
    showReferences();
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
  opening.addEventListener("change", showReferences);
  for (const option of presets) option.addEventListener("change", selectVisit);
  selectVisit();
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

