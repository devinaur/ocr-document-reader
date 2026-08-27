// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewBox = document.getElementById('previewBox');
const fileNameEl = document.getElementById('fileName');
const fileSizeEl = document.getElementById('fileSize');
const fileIconEl = document.getElementById('fileIcon');
const changeFileBtn = document.getElementById('changeFileBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const loadingState = document.getElementById('loadingState');
const loadingStatus = document.getElementById('loadingStatus');

const emptyResults = document.getElementById('emptyResults');
const resultsContent = document.getElementById('resultsContent');

const resDocType = document.getElementById('resDocType');
const resTitle = document.getElementById('resTitle');
const resDate = document.getElementById('resDate');
const resSummary = document.getElementById('resSummary');
const resEntities = document.getElementById('resEntities');
const resRawText = document.getElementById('resRawText');

let selectedFile = null;

// Event Listeners for File Selection
browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener('click', () => {
  if (!selectedFile) fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFileSelected(e.target.files[0]);
  }
});

// Drag & Drop Listeners
['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (dt.files && dt.files.length > 0) {
    handleFileSelected(dt.files[0]);
  }
});

changeFileBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  resetFileSelection();
});

function handleFileSelected(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatBytes(file.size);
  
  if (file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf') {
    fileIconEl.textContent = '📑';
  } else {
    fileIconEl.textContent = '🖼️';
  }

  dropZone.classList.add('hidden');
  previewBox.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

function resetFileSelection() {
  selectedFile = null;
  fileInput.value = '';
  dropZone.classList.remove('hidden');
  previewBox.classList.add('hidden');
  analyzeBtn.disabled = true;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Analyze Button Click Handler
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Show loading state
  analyzeBtn.disabled = true;
  loadingState.classList.remove('hidden');
  loadingStatus.textContent = 'Step 1/2: Running PaddleOCR Text Detection...';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    // Send POST request to FastAPI backend (/api/analyze)
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    loadingStatus.textContent = 'Step 2/2: Structuring with Llama 3.2 3B via Ollama...';

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to analyze document.');
    }

    const data = await response.json();
    renderResults(data);

  } catch (err) {
    alert(`❌ Analysis Error: ${err.message}`);
  } finally {
    loadingState.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
});

function formatEntityValue(val) {
  if (val === null || val === undefined) return '<span style="color:#64748b">N/A</span>';
  if (typeof val === 'object') {
    if (Array.isArray(val)) {
      return val.map((item) => `<div style="margin-bottom:0.35rem; padding-left:0.5rem; border-left:2px solid #6366f1">${formatEntityValue(item)}</div>`).join('');
    } else {
      return Object.entries(val)
        .map(([k, v]) => `<div style="margin-top:0.2rem"><strong style="color:#a5b4fc; text-transform:capitalize">${k.replace(/_/g, ' ')}:</strong> ${typeof v === 'object' ? formatEntityValue(v) : String(v)}</div>`)
        .join('');
    }
  }
  return String(val);
}

function renderResults(data) {
  const doc = data.document_result;

  resDocType.textContent = doc.document_type || 'UNKNOWN';
  resTitle.textContent = doc.title || 'N/A';
  resDate.textContent = doc.date || 'N/A';
  resSummary.textContent = doc.summary || 'No summary available.';

  // Render Entities
  resEntities.innerHTML = '';
  const entities = doc.entities || {};
  
  if (Object.keys(entities).length === 0) {
    resEntities.innerHTML = '<div class="entity-card"><div class="entity-key">Entities</div><div class="entity-value">None extracted</div></div>';
  } else {
    for (const [key, val] of Object.entries(entities)) {
      const card = document.createElement('div');
      card.className = 'entity-card';
      
      const keyEl = document.createElement('div');
      keyEl.className = 'entity-key';
      keyEl.textContent = key.replace(/_/g, ' ');

      const valEl = document.createElement('div');
      valEl.className = 'entity-value';
      valEl.innerHTML = formatEntityValue(val);

      card.appendChild(keyEl);
      card.appendChild(valEl);
      resEntities.appendChild(card);
    }
  }

  // Render Raw Text
  resRawText.textContent = data.raw_text || '(No text extracted)';

  emptyResults.classList.add('hidden');
  resultsContent.classList.remove('hidden');
}
