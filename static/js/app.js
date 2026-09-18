// Deep Learning Time Series Forecasting Frontend Logic

let chartInstance = null;
let currentHistoryData = [];
let metadata = null;

const MODEL_COLORS = {
  'LSTM': { stroke: '#6366f1', fill: 'rgba(99, 102, 241, 0.15)' },
  'CNN-LSTM': { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.15)' },
  'CNN': { stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.15)' },
  'MLP': { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.15)' }
};

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await initializeApp();
});

async function initializeApp() {
  try {
    const res = await fetch('/api/metadata');
    metadata = await res.json();
    populateSelects();
    await updateDashboard();
  } catch (err) {
    console.error("Initialization failed:", err);
    showError("Could not connect to API server. Ensure FastAPI backend is running.");
  }
}

function populateSelects() {
  const storeSelect = document.getElementById('storeSelect');
  const itemSelect = document.getElementById('itemSelect');
  
  storeSelect.innerHTML = '';
  metadata.stores.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = `Store ${s}`;
    storeSelect.appendChild(opt);
  });
  
  itemSelect.innerHTML = '';
  metadata.items.forEach(i => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Item ${i}`;
    itemSelect.appendChild(opt);
  });
}

function setupEventListeners() {
  // Store & Item change
  document.getElementById('storeSelect').addEventListener('change', () => updateDashboard());
  document.getElementById('itemSelect').addEventListener('change', () => updateDashboard());

  // Model Selection Cards
  const modelCards = document.querySelectorAll('.model-card');
  modelCards.forEach(card => {
    card.addEventListener('click', () => {
      modelCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      updateDashboard();
    });
  });

  // Horizon Slider
  const slider = document.getElementById('horizonSlider');
  const sliderVal = document.getElementById('horizonValue');
  slider.addEventListener('input', (e) => {
    sliderVal.textContent = `${e.target.value} Days`;
  });
  slider.addEventListener('change', () => updateDashboard());

  // Compare All Models Button
  document.getElementById('btnCompare').addEventListener('click', runComparison);

  // Modal handlers
  const modal = document.getElementById('uploadModal');
  document.getElementById('btnOpenUpload').addEventListener('click', () => modal.classList.add('active'));
  document.getElementById('btnCloseUpload').addEventListener('click', () => modal.classList.remove('active'));
  
  // File Upload
  const fileInput = document.getElementById('csvFileInput');
  const dropzone = document.getElementById('uploadDropzone');
  
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileUpload);
}

function getSelectedModel() {
  const activeCard = document.querySelector('.model-card.active');
  return activeCard ? activeCard.dataset.model : 'LSTM';
}

async function updateDashboard() {
  setLoading(true);
  hideError();
  const storeId = parseInt(document.getElementById('storeSelect').value, 10);
  const itemId = parseInt(document.getElementById('itemSelect').value, 10);
  const horizon = parseInt(document.getElementById('horizonSlider').value, 10);
  const model = getSelectedModel();

  try {
    // 1. Fetch History
    const histRes = await fetch(`/api/history?store_id=${storeId}&item_id=${itemId}&days=90`);
    if (!histRes.ok) throw new Error("Failed to load historical data");
    const histData = await histRes.json();
    currentHistoryData = histData.data;

    // 2. Fetch Forecast
    const fcRes = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ store_id: storeId, item_id: itemId, model_name: model, horizon: horizon })
    });
    
    if (!fcRes.ok) {
      const err = await fcRes.json();
      throw new Error(err.detail || "Forecasting error");
    }
    const forecastData = await fcRes.json();

    // 3. Render Chart & KPIs
    renderSingleForecastChart(currentHistoryData, forecastData, model);
    updateKPIs(forecastData, model);

  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

async function runComparison() {
  setLoading(true);
  hideError();
  const storeId = parseInt(document.getElementById('storeSelect').value, 10);
  const itemId = parseInt(document.getElementById('itemSelect').value, 10);
  const horizon = parseInt(document.getElementById('horizonSlider').value, 10);

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ store_id: storeId, item_id: itemId, horizon: horizon })
    });
    if (!res.ok) throw new Error("Failed to run comparison");
    const data = await res.json();
    renderComparisonChart(currentHistoryData, data.comparisons);
    renderComparisonTable(data.comparisons);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function updateKPIs(forecastData, model) {
  const sum = forecastData.summary;
  document.getElementById('kpiTotal').textContent = sum.total_projected_sales.toLocaleString();
  document.getElementById('kpiAvg').textContent = `${sum.avg_daily_sales} / day`;
  document.getElementById('kpiPeak').textContent = sum.peak_day || 'N/A';

  const modelMeta = metadata?.models?.find(m => m.id === model);
  document.getElementById('kpiRmse').textContent = modelMeta ? `${modelMeta.val_rmse}` : '18.76';
}

function renderSingleForecastChart(history, forecastObj, modelName) {
  const ctx = document.getElementById('salesChart').getContext('2d');
  
  const histLabels = history.map(h => h.date);
  const histSales = history.map(h => h.sales);
  
  const fcLabels = forecastObj.forecast.map(f => f.date);
  const fcSales = forecastObj.forecast.map(f => f.sales);

  const allLabels = [...histLabels, ...fcLabels];

  // Pad arrays so forecast lines connect seamlessly to last history point
  const histPadded = [...histSales, ...Array(fcSales.length).fill(null)];
  const fcPadded = [...Array(histSales.length - 1).fill(null), histSales[histSales.length - 1], ...fcSales];

  if (chartInstance) chartInstance.destroy();

  const color = MODEL_COLORS[modelName] || MODEL_COLORS['LSTM'];

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Historical Sales',
          data: histPadded,
          borderColor: '#94a3b8',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.25,
          fill: false
        },
        {
          label: `Forecast (${modelName})`,
          data: fcPadded,
          borderColor: color.stroke,
          backgroundColor: color.fill,
          borderWidth: 2.5,
          borderDash: [5, 4],
          pointRadius: 1,
          pointHoverRadius: 5,
          pointBackgroundColor: color.stroke,
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: getChartOptions()
  });

  document.getElementById('comparisonWrapper').classList.add('hidden');
}

function renderComparisonChart(history, comparisons) {
  const ctx = document.getElementById('salesChart').getContext('2d');
  const histLabels = history.map(h => h.date);
  const histSales = history.map(h => h.sales);

  // Take labels from first successful model
  const firstModel = Object.values(comparisons).find(c => c.forecast);
  if (!firstModel) return;

  const fcLabels = firstModel.forecast.map(f => f.date);
  const allLabels = [...histLabels, ...fcLabels];

  const datasets = [
    {
      label: 'Historical Sales',
      data: [...histSales, ...Array(fcLabels.length).fill(null)],
      borderColor: '#94a3b8',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2
    }
  ];

  for (const [mName, mData] of Object.entries(comparisons)) {
    if (!mData.forecast) continue;
    const color = MODEL_COLORS[mName] || { stroke: '#fff' };
    const fcSales = mData.forecast.map(f => f.sales);
    const fcPadded = [...Array(histSales.length - 1).fill(null), histSales[histSales.length - 1], ...fcSales];

    datasets.push({
      label: `${mName} (RMSE: ${mData.val_rmse})`,
      data: fcPadded,
      borderColor: color.stroke,
      borderWidth: 2.2,
      pointRadius: 0,
      tension: 0.35,
      fill: false
    });
  }

  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: allLabels, datasets },
    options: getChartOptions()
  });
}

function renderComparisonTable(comparisons) {
  const wrapper = document.getElementById('comparisonWrapper');
  const tbody = document.getElementById('comparisonTableBody');
  tbody.innerHTML = '';

  for (const [mName, mData] of Object.entries(comparisons)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${mName}</strong></td>
      <td>${mData.description}</td>
      <td><span class="rmse-badge">${mData.val_rmse}</span></td>
      <td>${mData.summary ? mData.summary.total_projected_sales.toLocaleString() : 'N/A'}</td>
      <td>${mData.summary ? mData.summary.avg_daily_sales : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  }

  wrapper.classList.remove('hidden');
}

function getChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: {
        labels: {
          color: '#cbd5e1',
          font: { family: 'Inter', size: 12 }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        titleColor: '#f8fafc',
        bodyColor: '#e2e8f0',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        padding: 10,
        boxPadding: 4
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'Inter' } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.06)' },
        ticks: { color: '#64748b', font: { family: 'Inter' } }
      }
    }
  };
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', getSelectedModel());
  formData.append('horizon', parseInt(document.getElementById('horizonSlider').value, 10));

  setLoading(true);
  try {
    const res = await fetch('/api/upload-csv', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload forecast failed");
    }
    const data = await res.json();
    document.getElementById('uploadModal').classList.remove('active');
    renderCustomForecast(data);
  } catch (err) {
    alert("CSV Processing error: " + err.message);
  } finally {
    setLoading(false);
  }
}

function renderCustomForecast(data) {
  const result = data.result;
  const labels = result.forecast.map(f => `Step +${f.step}`);
  const values = result.forecast.map(f => f.sales);

  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById('salesChart').getContext('2d');

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: `Custom Forecast (${data.filename} - ${result.model})`,
        data: values,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.15)',
        fill: true,
        tension: 0.3
      }]
    },
    options: getChartOptions()
  });

  document.getElementById('kpiTotal').textContent = result.summary.total_projected_sales.toLocaleString();
  document.getElementById('kpiAvg').textContent = `${result.summary.avg_daily_sales} / step`;
  document.getElementById('kpiPeak').textContent = `Max: ${Math.max(...values)}`;
}

function setLoading(isLoading) {
  const spinner = document.getElementById('loadingSpinner');
  if (spinner) {
    if (isLoading) spinner.classList.remove('hidden');
    else spinner.classList.add('hidden');
  }
}

function showError(msg) {
  const alertBox = document.getElementById('errorAlert');
  if (alertBox) {
    alertBox.textContent = msg;
    alertBox.classList.remove('hidden');
  }
}

function hideError() {
  const alertBox = document.getElementById('errorAlert');
  if (alertBox) alertBox.classList.add('hidden');
}
