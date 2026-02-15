/**
 * Advanced Retrieval Evaluation - Frontend
 * Connects to FastAPI backend for execution log streaming and executive summary
 */

const API_BASE = window.location.origin;

const steps = ['loading', 'building', 'evaluating', 'summary'];
let currentStep = -1;

function getStepEl(step) {
  return document.getElementById(`step-${step}`);
}

function setStep(step) {
  const idx = steps.indexOf(step);
  if (idx < 0) return;
  steps.forEach((s, i) => {
    const el = getStepEl(s);
    el.classList.remove('active', 'complete');
    if (i < idx) el.classList.add('complete');
    else if (i === idx) el.classList.add('active');
  });
  currentStep = idx;
}

function addLogEntry(entry) {
  const container = document.getElementById('log-entries');
  const empty = document.getElementById('log-empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `log-entry ${entry.level || 'info'}`;
  div.innerHTML = `
    <span class="timestamp">${entry.timestamp}</span>
    <span class="message">${escapeHtml(entry.message)}</span>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;

  const countEl = document.getElementById('log-count');
  const count = container.querySelectorAll('.log-entry').length;
  countEl.textContent = `(${count})`;

  // Infer workflow step from log message
  const msg = (entry.message || '').toLowerCase();
  if (msg.includes('loading') || msg.includes('loaded')) setStep('loading');
  else if (msg.includes('retriever') && msg.includes('building')) setStep('building');
  else if (msg.includes('evaluating') || msg.includes('ragas')) setStep('evaluating');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

const SUMMARY_PLACEHOLDER = 'High-level summary of retriever performance: which strategy (naive, BM25, contextual compression, multi-query, parent document, ensemble) performs best on context recall, entity recall, and noise sensitivity. Click "Run Evaluation" or "Load Previous Results" to populate.';

function setExecutiveSummary(text) {
  const el = document.getElementById('executive-summary-text');
  el.textContent = text || SUMMARY_PLACEHOLDER;
  el.classList.toggle('summary-placeholder', !text || text === SUMMARY_PLACEHOLDER);
}

function ensureRetrieverTableHeader() {
  const container = document.getElementById('retriever-results');
  if (!container.querySelector('.retriever-header')) {
    container.innerHTML = '';
    const thead = document.createElement('div');
    thead.className = 'retriever-header';
    thead.innerHTML = `
      <span class="col-retriever">Retriever</span>
      <span class="col-metric">Context Recall</span>
      <span class="col-metric">Entity Recall</span>
      <span class="col-metric">Noise Sensitivity</span>
      <span class="col-metric">Latency (s)</span>
    `;
    container.appendChild(thead);
  }
}

function appendRetrieverResult(r) {
  ensureRetrieverTableHeader();
  const container = document.getElementById('retriever-results');
  const row = document.createElement('div');
  row.className = 'retriever-row';
  row.innerHTML = `
    <span class="name">${escapeHtml(r.retriever)}</span>
    <span class="metric"><strong>${((r.context_recall ?? 0) * 100).toFixed(2)}%</strong></span>
    <span class="metric"><strong>${((r.context_entity_recall ?? 0) * 100).toFixed(2)}%</strong></span>
    <span class="metric"><strong>${((r.noise_sensitivity ?? 0) * 100).toFixed(2)}%</strong></span>
    <span class="metric"><strong>${(r.latency_seconds ?? 0).toFixed(2)}</strong></span>
  `;
  container.appendChild(row);
}

function renderRetrieverResults(results) {
  const container = document.getElementById('retriever-results');
  container.innerHTML = '';

  if (!results || !results.length) {
    container.innerHTML = '<p class="results-placeholder">No results available.</p>';
    return;
  }

  ensureRetrieverTableHeader();
  results.forEach(r => appendRetrieverResult(r));

  document.getElementById('rca-status').textContent = 'COMPLETE';
  document.getElementById('rca-status').className = 'status-pill complete';
  setStep('summary');
}

function setDeepDiveData(data) {
  const content = document.getElementById('deep-dive-content');
  const pre = document.getElementById('deep-dive-raw');
  const btn = document.getElementById('show-details-btn');
  if (data) {
    try {
      pre.textContent = JSON.stringify(data, null, 2);
      content.classList.remove('hidden');
      btn.textContent = '▼ Hide Raw JSON';
    } catch (e) {
      pre.textContent = 'Could not serialize: ' + String(e);
    }
  } else {
    pre.textContent = 'No data yet. Click "Load Previous Results" or run evaluation.';
  }
}

function showDeepDive(data) {
  setDeepDiveData(data);
}

document.getElementById('show-details-btn').addEventListener('click', () => {
  const content = document.getElementById('deep-dive-content');
  const btn = document.getElementById('show-details-btn');
  if (content.classList.contains('hidden')) {
    content.classList.remove('hidden');
    btn.textContent = '▼ Hide Raw JSON';
  } else {
    content.classList.add('hidden');
    btn.textContent = '► Show Raw JSON';
  }
});

document.getElementById('log-toggle').addEventListener('click', () => {
  const body = document.getElementById('log-body');
  const arrow = document.querySelector('.collapse-arrow');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
  arrow.style.transform = body.style.display === 'none' ? 'rotate(-90deg)' : 'rotate(0)';
});

async function runEvaluation() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;

  document.getElementById('log-entries').innerHTML = '';
  document.getElementById('log-count').textContent = '(0)';
  addLogEntry({ timestamp: '--:--:-- --', message: 'Connecting to evaluation stream...', level: 'info' });
  setStep('loading');

  // Reset Retriever Comparison and Raw Evaluation Data - in sync with run
  const resultsContainer = document.getElementById('retriever-results');
  resultsContainer.innerHTML = '';
  ensureRetrieverTableHeader();
  document.getElementById('rca-status').textContent = 'RUNNING';
  document.getElementById('rca-status').className = 'status-pill running';

  setDeepDiveData({ status: 'running', message: 'Evaluation in progress...', results_so_far: [] });
  setExecutiveSummary('Evaluation in progress... Results will appear here when complete.');

  const accumulatedResults = [];

  try {
    const es = new EventSource(`${API_BASE}/api/run/stream`);

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === 'log') {
          addLogEntry(data.payload);
        } else if (data.type === 'retriever_result') {
          accumulatedResults.push(data.payload);
          appendRetrieverResult(data.payload);
          setDeepDiveData({ status: 'running', results_so_far: [...accumulatedResults] });
        } else if (data.type === 'complete') {
          es.close();
          const payload = data.payload;
          if (payload && payload.success) {
            setExecutiveSummary(payload.executive_summary || 'Evaluation completed.');
            if (payload.results_summary) {
              renderRetrieverResults(payload.results_summary);
            }
            setDeepDiveData(payload);
          } else if (payload && payload.error) {
            addLogEntry({ timestamp: '--', message: `Error: ${payload.error}`, level: 'error' });
            setExecutiveSummary(`Evaluation failed: ${payload.error}`);
            setDeepDiveData({ error: payload.error, results_so_far: accumulatedResults });
          } else {
            setDeepDiveData(payload);
          }
          btn.disabled = false;
        } else if (data.type === 'error') {
          addLogEntry({ timestamp: '--', message: `Error: ${data.payload?.error || 'Unknown'}`, level: 'error' });
          es.close();
          btn.disabled = false;
        }
      } catch (e) {
        console.error('Parse error', e);
      }
    };

    es.onerror = () => {
      es.close();
      addLogEntry({ timestamp: '--', message: 'Stream disconnected.', level: 'error' });
      setExecutiveSummary('Evaluation was interrupted (stream disconnected).');
      setDeepDiveData({ error: 'Stream disconnected', results_so_far: accumulatedResults });
      btn.disabled = false;
    };
  } catch (err) {
    setExecutiveSummary(`Failed to start evaluation: ${err.message}`);
    addLogEntry({ timestamp: '--', message: `Failed to start: ${err.message}`, level: 'error' });
    btn.disabled = false;
  }
}

async function loadPreviousResults() {
  try {
    const [resResp, sumResp] = await Promise.all([
      fetch(`${API_BASE}/api/results`),
      fetch(`${API_BASE}/api/executive-summary`),
    ]);

    const resData = await resResp.json();
    const sumData = await sumResp.json();

    if (resData.results && resData.results.length) {
      setExecutiveSummary(sumData.summary || `Loaded ${resData.results.length} retriever results from previous run.`);
      renderRetrieverResults(resData.results);
      showDeepDive({ results: resData.results, summary: sumData.summary });
      setStep('summary');
      document.querySelectorAll('.workflow-step').forEach(el => el.classList.add('complete'));
      addLogEntry({
        timestamp: '--',
        message: `Loaded ${resData.results.length} retriever results from previous run.`,
        level: 'info',
      });
    } else {
      setExecutiveSummary(SUMMARY_PLACEHOLDER);
      setDeepDiveData(null);
      addLogEntry({
        timestamp: '--',
        message: 'No previous results found. Run evaluation first.',
        level: 'warning',
      });
    }
  } catch (err) {
    setExecutiveSummary(`Load failed: ${err.message}`);
    setDeepDiveData({ error: err.message });
    addLogEntry({ timestamp: '--', message: `Load failed: ${err.message}`, level: 'error' });
  }
}

document.getElementById('run-btn').addEventListener('click', runEvaluation);
document.getElementById('load-results-btn').addEventListener('click', loadPreviousResults);

// Auto-load saved results on page load for quick UI preview (no need to run long evaluation)
loadPreviousResults();
