/**
 * CyberShield AI - Real-Time Threat Detection Demo
 * Frontend for Techfest Demonstration
 */

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Global state
let currentPrediction = null;
let predictionHistory = [];

// ========== Initialization ==========
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
    setupEventListeners();
    setupFileUpload();
});

async function initializeApp() {
    // Check API and model status
    await checkAPIHealth();
    await loadStatistics();
    await loadStatistics();
    await loadRecentPredictions();

    // Start polling simulation status immediately
    simulationPollInterval = setInterval(pollSimulationStatus, 1000);

    // Setup navigation
    setupSmoothScroll();
}

// ========== API Communication ==========
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API Error:', error);
        showError('Unable to connect to backend. Please ensure the server is running on port 5000');
        return null;
    }
}

async function checkAPIHealth() {
    const data = await apiCall('/health');

    const apiStatus = document.getElementById('apiStatus');
    const modelStatus = document.getElementById('modelStatus');

    if (data && data.status === 'healthy') {
        apiStatus.textContent = 'Connected';
        apiStatus.className = 'status-badge success';

        if (data.model_loaded) {
            modelStatus.textContent = 'Loaded';
            modelStatus.className = 'status-badge success';
        } else {
            modelStatus.textContent = 'Not Found';
            modelStatus.className = 'status-badge warning';
        }
    } else {
        apiStatus.textContent = 'Disconnected';
        apiStatus.className = 'status-badge error';
        modelStatus.textContent = 'Unknown';
        modelStatus.className = 'status-badge';
    }
}

// ========== File Upload & Prediction ==========
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await uploadAndPredict(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await uploadAndPredict(e.target.files[0]);
        }
    });
}

async function uploadAndPredict(file) {
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        return;
    }

    showLoading(`Analyzing ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload-predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        hideLoading();

        if (data.success) {
            currentPrediction = data.results;
            displayPredictionResults(data.results, file.name);
            showSuccess(`✓ Analyzed ${data.results.total_samples} samples in ${data.results.processing_time_seconds.toFixed(2)}s`);

            // Update statistics
            await loadStatistics();
            await loadRecentPredictions();
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        hideLoading();
        showError('Upload failed: ' + error.message);
    }
}

function displayPredictionResults(results, filename) {
    const resultsArea = document.getElementById('resultsArea');
    const resultsContainer = document.getElementById('detectionResults');

    resultsArea.style.display = 'block';

    // Performance section
    let html = `
        <div class="result-header">
            <h4>📊 Analysis Results</h4>
            <span class="file-badge">${filename}</span>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">${formatNumber(results.total_samples)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Processing Time</div>
                <div class="metric-value">${results.processing_time_seconds.toFixed(2)}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value">${formatNumber(Math.round(results.samples_per_second))}/s</div>
            </div>
    `;

    // Add performance metrics if available
    if (results.performance_metrics) {
        const pm = results.performance_metrics;
        html += `
            <div class="metric-card highlight">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">${(pm.accuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    html += `</div>`;

    // Threat distribution
    html += `
        <div class="threat-distribution">
            <h4>🎯 Threat Distribution</h4>
            <div class="distribution-grid">
    `;

    const sortedDistribution = Object.entries(results.prediction_distribution)
        .sort((a, b) => b[1] - a[1]);

    sortedDistribution.forEach(([threat, count]) => {
        const percentage = ((count / results.total_samples) * 100).toFixed(1);
        const isBenign = threat.toUpperCase() === 'BENIGN';

        html += `
            <div class="distribution-item ${isBenign ? 'benign' : 'threat'}">
                <div class="distribution-header">
                    <span class="threat-name">${threat}</span>
                    <span class="threat-count">${formatNumber(count)}</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: ${percentage}%;"></div>
                </div>
                <div class="threat-percentage">${percentage}%</div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    // Performance metrics details if available
    if (results.performance_metrics) {
        const pm = results.performance_metrics;
        html += `
            <div class="performance-section">
                <h4>📈 Model Performance</h4>
                <div class="performance-grid">
                    <div class="perf-item">
                        <span class="perf-label">Precision</span>
                        <span class="perf-value">${(pm.precision * 100).toFixed(2)}%</span>
                    </div>
                    <div class="perf-item">
                        <span class="perf-label">Recall</span>
                        <span class="perf-value">${(pm.recall * 100).toFixed(2)}%</span>
                    </div>
                    <div class="perf-item">
                        <span class="perf-label">F1-Score</span>
                        <span class="perf-value">${(pm.f1_score * 100).toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    resultsContainer.innerHTML = html;

    // Create chart
    createThreatChart(results.prediction_distribution);
}

function clearResults() {
    document.getElementById('resultsArea').style.display = 'none';
    document.getElementById('detectionResults').innerHTML = '';
    document.getElementById('fileInput').value = '';
}

// ========== Charts ==========
function createThreatChart(distribution) {
    const canvas = document.getElementById('threatChart');
    if (!canvas) return;

    // Destroy existing chart
    if (window.threatChartInstance) {
        window.threatChartInstance.destroy();
    }

    const labels = Object.keys(distribution);
    const data = Object.values(distribution);

    // Generate colors
    const colors = labels.map(label =>
        label.toUpperCase() === 'BENIGN' ?
            'rgba(16, 185, 129, 0.8)' :
            'rgba(239, 68, 68, 0.8)'
    );

    window.threatChartInstance = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#b4b9d0',
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    });
}

// ========== Statistics ==========
async function loadStatistics() {
    const data = await apiCall('/stats');

    if (data && data.success) {
        updateStatisticsDisplay(data.statistics);
    }
}

function updateStatisticsDisplay(stats) {
    // Update dashboard stats
    document.getElementById('totalSamples').textContent = formatNumber(stats.total_samples_processed || 0);
    document.getElementById('totalPredictions').textContent = stats.total_predictions || 0;

    if (stats.average_processing_time) {
        document.getElementById('avgTime').textContent = stats.average_processing_time.toFixed(2) + 's';
    }

    // Update threat distribution chart if available
    if (stats.threat_distribution && Object.keys(stats.threat_distribution).length > 0) {
        createStatsChart(stats.threat_distribution);
    }
}

function createStatsChart(distribution) {
    const canvas = document.getElementById('statsChart');
    if (!canvas) return;

    if (window.statsChartInstance) {
        window.statsChartInstance.destroy();
    }

    const labels = Object.keys(distribution);
    const data = Object.values(distribution);

    window.statsChartInstance = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Detections',
                data: data,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderWidth: 0,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#b4b9d0'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#b4b9d0',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// ========== Recent Predictions ==========
async function loadRecentPredictions() {
    const data = await apiCall('/recent-predictions');

    if (data && data.success && data.predictions.length > 0) {
        displayRecentPredictions(data.predictions);
    }
}

function displayRecentPredictions(predictions) {
    const container = document.getElementById('recentPredictions');
    if (!container) return;

    const html = predictions.slice(0, 5).map(pred => `
        <div class="history-item">
            <div class="history-header">
                <span class="history-file">${pred.filename}</span>
                <span class="history-time">${new Date(pred.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="history-stats">
                <span>${formatNumber(pred.total_samples)} samples</span>
                <span>${pred.processing_time.toFixed(2)}s</span>
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

// ========== Navigation ==========
function setupSmoothScroll() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            const targetId = link.getAttribute('href');
            scrollToSection(targetId.substring(1));

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const offset = 80;
        const bodyRect = document.body.getBoundingClientRect().top;
        const sectionRect = section.getBoundingClientRect().top;
        const sectionPosition = sectionRect - bodyRect;
        const offsetPosition = sectionPosition - offset;

        window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
        });
    }
}

function setupEventListeners() {
    window.addEventListener('scroll', () => {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(26, 32, 53, 0.95)';
        } else {
            navbar.style.background = 'rgba(26, 32, 53, 0.7)';
        }
    });
}

// ========== UI Utilities ==========
function showLoading(text = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = text;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
}

function showSuccess(message) {
    // Simple alert for now - can be replaced with custom notification
    console.log('✓ Success:', message);
    // Could add a toast notification here
}

function showError(message) {
    alert('✗ Error: ' + message);
}

// ========== Formatting Utilities ==========
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
// ========== Simulation Logic ==========
let simulations = {
    'A': { running: false, interval: null },
    'B': { running: false, interval: null }
};

let simulationPollInterval = null;

async function toggleSimulation(networkId, type) {
    const isRunning = simulations[networkId].running;
    const btn = document.getElementById(`btnStart${networkId}`);

    if (isRunning) {
        // Stop simulation
        const result = await apiCall('/simulation/stop', 'POST', { network_id: networkId });
        if (result && result.success) {
            simulations[networkId].running = false;
            updateSimulationUI(networkId, false);
            addLog(networkId, 'system', 'Simulation stopped.');
        }
    } else {
        // Start simulation
        const result = await apiCall('/simulation/start', 'POST', {
            network_id: networkId,
            type: type
        });

        if (result && result.success) {
            simulations[networkId].running = true;
            updateSimulationUI(networkId, true);
            addLog(networkId, 'system', `Starting ${type} traffic simulation...`);

            // Start polling if not active
            if (!simulationPollInterval) {
                simulationPollInterval = setInterval(pollSimulationStatus, 1000);
            }
        }
    }
}

function updateSimulationUI(networkId, isRunning) {
    const btn = document.getElementById(`btnStart${networkId}`);
    const statusBadge = document.getElementById(`status${networkId}`);

    if (isRunning) {
        btn.textContent = '⏹ Stop Simulation';
        btn.classList.replace('btn-primary', 'btn-secondary');
        btn.classList.replace('btn-danger', 'btn-secondary');
        statusBadge.className = networkId === 'A' ? 'status-badge success pulsing' : 'status-badge danger pulsing';
        statusBadge.textContent = 'ACTIVE';
    } else {
        const isSecure = networkId === 'A';
        btn.textContent = isSecure ? '▶ Follow Normal Traffic' : '⚠ Simulate DDoS Attack';
        btn.className = `btn btn-sm ${isSecure ? 'btn-primary' : 'btn-danger'}`;
        statusBadge.className = isSecure ? 'status-badge success' : 'status-badge neutral';
        statusBadge.textContent = isSecure ? 'SECURE' : 'IDLE';
    }
}

async function pollSimulationStatus() {
    const data = await apiCall('/simulation/status');

    if (data && data.success) {
        let activeCount = 0;

        Object.entries(data.status).forEach(([netId, status]) => {
            if (status.running) {
                activeCount++;
                updateMonitor(netId, status);
            } else if (simulations[netId].running) {
                // Backend stopped, update frontend
                simulations[netId].running = false;
                updateSimulationUI(netId, false);
            }
        });

        if (activeCount === 0 && simulationPollInterval) {
            clearInterval(simulationPollInterval);
            simulationPollInterval = null;
        }
    }
}

function updateMonitor(networkId, status) {
    if (!status.latest_result) return;

    const result = status.latest_result;
    const trafficEl = document.getElementById(`traffic${networkId}`);
    const threatsEl = document.getElementById(`threats${networkId}`);

    // Update stats
    trafficEl.textContent = `${Math.round(result.samples_per_second)} p/s`;

    // Count threats in this batch
    const threats = result.predictions.filter(p => p !== 'BENIGN').length;
    threatsEl.textContent = status.stats.attacks;

    // Add log entry
    const isAttack = threats > 0;
    const type = isAttack ? 'attack' : 'benign';
    const msg = isAttack
        ? `[ALERT] ${result.predictions[0]} detected! Conf: ${(result.confidence_scores[0] * 100).toFixed(1)}%`
        : `[INFO] Normal traffic processed.`;

    addLog(networkId, type, msg);

    // Flash status if under attack
    if (isAttack) {
        const badge = document.getElementById(`status${networkId}`);
        badge.textContent = 'BREACHED';
        badge.className = 'status-badge danger pulsing';
    } else if (networkId === 'A') {
        const badge = document.getElementById(`status${networkId}`);
        badge.textContent = 'SECURE';
        badge.className = 'status-badge success pulsing';
    }
}

function addLog(networkId, type, message) {
    const logContainer = document.getElementById(`log${networkId}`);
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `> ${message}`;

    logContainer.appendChild(entry);

    // Auto scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;

    // Limit log size
    if (logContainer.children.length > 20) {
        logContainer.removeChild(logContainer.firstChild);
    }
}
