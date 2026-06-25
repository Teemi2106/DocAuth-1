/* DocVerify — frontend JS */

// ── File upload handling ──────────────────────────────────────────────────────

function handleFileSelect(file) {
  if (!file) return;

  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadLabel = document.getElementById('uploadLabel');
  const uploadSub   = document.getElementById('uploadSub');
  const uploadIcon  = document.getElementById('uploadIcon');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg  = document.getElementById('previewImg');
  const previewBadge = document.getElementById('previewBadge');

  if (!analyzeBtn) return;

  analyzeBtn.disabled = false;
  previewBadge.textContent = file.name.length > 24
    ? file.name.slice(0, 22) + '…' + file.name.slice(-6)
    : file.name;

  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      uploadIcon.classList.add('hidden');
      previewWrap.classList.remove('hidden');
      uploadLabel.textContent = 'File ready for analysis';
      uploadSub.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    };
    reader.readAsDataURL(file);
  } else {
    // PDF or non-image: show icon only
    uploadIcon.classList.remove('hidden');
    previewWrap.classList.add('hidden');
    uploadLabel.textContent = `Selected: ${file.name}`;
    uploadSub.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  }
}

function clearUpload() {
  const fileInput   = document.getElementById('fileInput');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  const uploadLabel = document.getElementById('uploadLabel');
  const uploadSub   = document.getElementById('uploadSub');
  const uploadIcon  = document.getElementById('uploadIcon');
  const previewWrap = document.getElementById('previewWrap');

  if (!fileInput) return;

  fileInput.value = '';
  if (analyzeBtn)  analyzeBtn.disabled = true;
  if (uploadIcon)  uploadIcon.classList.remove('hidden');
  if (previewWrap) previewWrap.classList.add('hidden');
  if (uploadLabel) {
    uploadLabel.innerHTML = '<strong>Click to upload</strong> or drag and drop your document';
  }
  if (uploadSub) uploadSub.textContent = 'PNG, JPG, TIFF, BMP, PDF — max 16 MB';
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────

(function initDragDrop() {
  const zone = document.getElementById('dropZone');
  if (!zone) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      // Assign to the file input so the form submits it
      const dt = new DataTransfer();
      dt.items.add(file);
      document.getElementById('fileInput').files = dt.files;
      handleFileSelect(file);
    }
  });
})();
