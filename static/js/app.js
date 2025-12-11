/**
 * Bitcoin Market Analysis - Frontend Application
 */

class BTCAnalyzer {
    constructor() {
        this.socket = null;
        this.mainChart = null;
        this.indicatorChart = null;
        this.candlestickSeries = null;
        this.volumeSeries = null;
        this.indicatorSeries = {};
        this.lineSeries = {};
        this.markers = [];

        // Load persisted state from sessionStorage (survives reconnects, cleared on tab close)
        this.currentTimeframe = this.loadFromSession('currentTimeframe', '15m');
        this.currentIndicator = this.loadFromSession('currentIndicator', 'macd');
        this.soundEnabled = this.loadFromSession('soundEnabled', true);
        this.lastPrice = 0;  // Current price for display
        this.initialPrice = this.loadFromSession('initialPrice', null);  // Price when session started

        // Track if user has interacted with the chart (zoomed/scrolled)
        this.userHasInteracted = false;
        this.isInitialLoad = true;
        this.hasLoadedData = false;  // Track if we've loaded data at least once

        this.init();
    }

    // ==================== Session Storage ====================

    /**
     * Save a value to sessionStorage (persists across reconnects, cleared when tab closes)
     */
    saveToSession(key, value) {
        try {
            sessionStorage.setItem(`btc_analyzer_${key}`, JSON.stringify(value));
        } catch (e) {
            console.warn('Failed to save to sessionStorage:', e);
        }
    }

    /**
     * Load a value from sessionStorage
     */
    loadFromSession(key, defaultValue) {
        try {
            const stored = sessionStorage.getItem(`btc_analyzer_${key}`);
            if (stored !== null) {
                return JSON.parse(stored);
            }
        } catch (e) {
            console.warn('Failed to load from sessionStorage:', e);
        }
        return defaultValue;
    }

    init() {
        this.setupSocket();
        this.setupCharts();
        this.setupEventListeners();
        this.loadInitialData();

        // Apply saved indicator selection after charts are ready
        if (this.currentIndicator !== 'macd') {
            this.switchIndicator(this.currentIndicator);
        }
    }

    // ==================== Socket Connection ====================

    setupSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus('connected');
            // Notify backend of current timeframe on connect
            this.socket.emit('set_timeframe', { timeframe: this.currentTimeframe });
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('analysis_update', (data) => {
            this.handleAnalysisUpdate(data);
        });
    }

    updateConnectionStatus(status) {
        const indicator = document.getElementById('connection-status');
        const text = indicator.querySelector('.status-text');

        indicator.className = 'status-indicator ' + status;

        if (status === 'connected') {
            text.textContent = 'Live';
        } else if (status === 'disconnected') {
            text.textContent = 'Disconnected';
        } else {
            text.textContent = 'Connecting...';
        }
    }

    // ==================== Charts Setup ====================

    setupCharts() {
        this.setupMainChart();
        this.setupIndicatorChart();
    }

    setupMainChart() {
        const container = document.getElementById('main-chart');
        if (!container) return;

        this.mainChart = LightweightCharts.createChart(container, {
            layout: {
                background: { type: 'solid', color: '#0d1117' },
                textColor: '#8b949e',
            },
            grid: {
                vertLines: { color: '#21262d' },
                horzLines: { color: '#21262d' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: '#30363d',
            },
            timeScale: {
                borderColor: '#30363d',
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Candlestick series
        this.candlestickSeries = this.mainChart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        // EMA lines
        this.lineSeries.ema_fast = this.mainChart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 1,
            title: 'EMA 9',
        });

        this.lineSeries.ema_slow = this.mainChart.addLineSeries({
            color: '#3b82f6',
            lineWidth: 1,
            title: 'EMA 21',
        });

        // Bollinger Bands
        this.lineSeries.bb_upper = this.mainChart.addLineSeries({
            color: '#8b5cf6',
            lineWidth: 1,
            lineStyle: 2,
        });

        this.lineSeries.bb_lower = this.mainChart.addLineSeries({
            color: '#8b5cf6',
            lineWidth: 1,
            lineStyle: 2,
        });

        // Handle resize
        window.addEventListener('resize', () => {
            this.mainChart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        });

        // Initial size
        this.mainChart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight,
        });

        // Track user interaction with the chart
        this.mainChart.timeScale().subscribeVisibleTimeRangeChange(() => {
            // Only set userHasInteracted after initial load is complete
            if (!this.isInitialLoad) {
                this.userHasInteracted = true;
            }
        });
    }

    setupIndicatorChart() {
        const container = document.getElementById('indicator-chart');
        if (!container) return;

        this.indicatorChart = LightweightCharts.createChart(container, {
            layout: {
                background: { type: 'solid', color: '#161b22' },
                textColor: '#8b949e',
            },
            grid: {
                vertLines: { color: '#21262d' },
                horzLines: { color: '#21262d' },
            },
            rightPriceScale: {
                borderColor: '#30363d',
            },
            timeScale: {
                borderColor: '#30363d',
                visible: false,
            },
            height: 140,
        });

        // MACD series
        this.indicatorSeries.macd_line = this.indicatorChart.addLineSeries({
            color: '#3b82f6',
            lineWidth: 2,
            title: 'MACD',
        });

        this.indicatorSeries.macd_signal = this.indicatorChart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 1,
            title: 'Signal',
        });

        this.indicatorSeries.macd_histogram = this.indicatorChart.addHistogramSeries({
            color: '#22c55e',
        });

        // RSI series (initially hidden)
        this.indicatorSeries.rsi = this.indicatorChart.addLineSeries({
            color: '#a855f7',
            lineWidth: 2,
            title: 'RSI',
            visible: false,
        });

        // Volume series (initially hidden)
        this.indicatorSeries.volume = this.indicatorChart.addHistogramSeries({
            color: '#6b7280',
            visible: false,
        });

        // Resize
        window.addEventListener('resize', () => {
            this.indicatorChart.applyOptions({
                width: container.clientWidth,
            });
        });

        this.indicatorChart.applyOptions({
            width: container.clientWidth,
        });
    }

    // ==================== Data Loading ====================

    async loadInitialData() {
        await this.loadChartData(this.currentTimeframe);
    }

    async loadChartData(timeframe, preserveView = false) {
        try {
            const response = await fetch(`/api/chart/${timeframe}`);
            const data = await response.json();

            if (data.error) {
                console.error('Chart data error:', data.error);
                return;
            }

            this.updateChartData(data, preserveView);
        } catch (error) {
            console.error('Failed to load chart data:', error);
        }
    }

    updateChartData(data, preserveView = false) {
        // Save current visible range if we need to preserve the view
        // Preserve if: explicitly requested AND (user has interacted OR we've loaded data before)
        let savedRange = null;
        if (preserveView && (this.userHasInteracted || this.hasLoadedData)) {
            savedRange = this.mainChart.timeScale().getVisibleLogicalRange();
        }

        // Update candlesticks
        if (data.candles && data.candles.length > 0) {
            this.candlestickSeries.setData(data.candles);

            // Update price display
            const lastCandle = data.candles[data.candles.length - 1];
            this.updatePriceDisplay(lastCandle.close);
        }

        // Update indicators
        if (data.indicators) {
            const ind = data.indicators;

            if (ind.ema_fast) this.lineSeries.ema_fast.setData(ind.ema_fast);
            if (ind.ema_slow) this.lineSeries.ema_slow.setData(ind.ema_slow);
            if (ind.bb_upper) this.lineSeries.bb_upper.setData(ind.bb_upper);
            if (ind.bb_lower) this.lineSeries.bb_lower.setData(ind.bb_lower);

            // MACD
            if (ind.macd_line) this.indicatorSeries.macd_line.setData(ind.macd_line);
            if (ind.macd_signal) this.indicatorSeries.macd_signal.setData(ind.macd_signal);
            if (ind.macd_histogram) this.indicatorSeries.macd_histogram.setData(ind.macd_histogram);

            // RSI
            if (ind.rsi) this.indicatorSeries.rsi.setData(ind.rsi);

            // Volume
            if (ind.volume) this.indicatorSeries.volume.setData(ind.volume);
        }

        // Update markers for patterns
        if (data.patterns && data.patterns.length > 0) {
            this.updatePatternMarkers(data.patterns);
        }

        // Restore view or fit content
        if (savedRange && preserveView) {
            // Restore the previous view position
            this.mainChart.timeScale().setVisibleLogicalRange(savedRange);
        } else if (!this.hasLoadedData) {
            // Only fit content on first ever load
            this.mainChart.timeScale().fitContent();
            // Mark initial load as complete after a short delay
            setTimeout(() => {
                this.isInitialLoad = false;
            }, 500);
        }
        // If we've loaded data before but not preserving, keep current view

        // Mark that we've loaded data at least once
        this.hasLoadedData = true;
    }

    updatePatternMarkers(patterns) {
        const markers = patterns.map(p => ({
            time: p.time,
            position: p.position,
            color: p.color,
            shape: p.shape,
            text: p.text,
        }));

        this.candlestickSeries.setMarkers(markers);
    }

    // ==================== Analysis Updates ====================

    handleAnalysisUpdate(data) {
        console.log('Analysis update:', data);

        // Update price
        if (data.price) {
            this.updatePriceDisplay(data.price);
        }

        // Update signal score
        if (data.signal) {
            this.updateSignalDisplay(data.signal);
        }

        // Update trade suggestion
        if (data.suggestion) {
            this.updateTradeDisplay(data.suggestion);
        }

        // Update signal history
        if (data.signal_history) {
            this.updateHistoryDisplay(data.signal_history);
        }

        // Reload chart data - preserve view if we've already loaded data once
        // This handles both regular updates AND reconnection scenarios
        this.loadChartData(this.currentTimeframe, this.hasLoadedData);

        // Handle notification
        if (data.should_notify && data.suggestion) {
            this.showAlert(data);
            this.playSound(data.suggestion.direction);
        }
    }

    updatePriceDisplay(price) {
        const priceEl = document.getElementById('current-price');
        const changeEl = document.getElementById('price-change');

        // Store initial price when session starts (first price received)
        if (this.initialPrice === null) {
            this.initialPrice = price;
            this.saveToSession('initialPrice', price);
        }

        if (priceEl) {
            priceEl.textContent = `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }

        // Calculate percentage change from initial session price (not last price)
        if (changeEl && this.initialPrice > 0) {
            const change = ((price - this.initialPrice) / this.initialPrice) * 100;
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeEl.className = 'price-change ' + (change >= 0 ? 'positive' : 'negative');
        }

        this.lastPrice = price;
    }

    updateSignalDisplay(signal) {
        // Score
        const scoreEl = document.getElementById('signal-score');
        const labelEl = document.getElementById('signal-label');
        const pointerEl = document.getElementById('gauge-pointer');

        const score = signal.combined_score;

        if (scoreEl) {
            scoreEl.textContent = (score >= 0 ? '+' : '') + score.toFixed(2);
            scoreEl.className = 'score-value ' + (score > 0.1 ? 'bullish' : score < -0.1 ? 'bearish' : 'neutral');
        }

        if (labelEl) {
            labelEl.textContent = signal.strength;
        }

        if (pointerEl) {
            // Convert -1 to 1 score to 0 to 100 percentage
            const position = ((score + 1) / 2) * 100;
            pointerEl.style.left = `${position}%`;
        }

        // Component scores from primary timeframe
        const primaryTf = Object.keys(signal.timeframes)[0];
        if (primaryTf && signal.timeframes[primaryTf]) {
            const tfData = signal.timeframes[primaryTf];

            this.updateComponentBar('candlestick', tfData.candlestick_score);
            this.updateComponentBar('chart', tfData.chart_score);
            this.updateComponentBar('indicator', tfData.indicator_score);

            // Update patterns list
            this.updatePatternsList(tfData);
        }

        // Alignment
        this.updateAlignment(signal.alignment, signal.timeframes);
    }

    updateComponentBar(component, score) {
        const bar = document.getElementById(`${component}-bar`);
        const value = document.getElementById(`${component}-score`);

        if (bar) {
            // Convert -1 to 1 to 0 to 100
            const width = Math.abs(score) * 50;
            bar.style.width = `${50 + (score > 0 ? width : -width)}%`;
            bar.className = 'comp-fill ' + (score > 0.1 ? 'bullish' : score < -0.1 ? 'bearish' : '');

            // Visual indication of direction
            if (score > 0) {
                bar.style.marginLeft = '50%';
                bar.style.width = `${width}%`;
            } else {
                bar.style.marginLeft = `${50 - width}%`;
                bar.style.width = `${width}%`;
            }
        }

        if (value) {
            value.textContent = (score >= 0 ? '+' : '') + score.toFixed(2);
        }
    }

    updatePatternsList(tfData) {
        const container = document.getElementById('patterns-list');
        if (!container) return;

        const allPatterns = [
            ...(tfData.patterns || []).map(p => ({ name: p, type: 'candle', bullish: true })),
            ...(tfData.chart_patterns || []).map(p => ({ name: p, type: 'chart', bullish: !p.toLowerCase().includes('bear') })),
            ...(tfData.key_signals || []).map(p => ({ name: p, type: 'indicator', bullish: !p.toLowerCase().includes('bear') })),
        ];

        if (allPatterns.length === 0) {
            container.innerHTML = '<div class="no-patterns">No patterns detected</div>';
            return;
        }

        container.innerHTML = allPatterns.slice(0, 10).map(p => `
            <div class="pattern-item">
                <span class="pattern-dot ${p.bullish ? 'bullish' : 'bearish'}"></span>
                <span class="pattern-name">${p.name}</span>
                <span class="pattern-type">${p.type}</span>
            </div>
        `).join('');
    }

    updateAlignment(alignment, timeframes) {
        const fillEl = document.getElementById('alignment-fill');
        const valueEl = document.getElementById('alignment-value');
        const signalsEl = document.getElementById('tf-signals');

        if (fillEl) {
            fillEl.style.width = `${alignment * 100}%`;
        }

        if (valueEl) {
            valueEl.textContent = `${Math.round(alignment * 100)}%`;
        }

        if (signalsEl && timeframes) {
            signalsEl.innerHTML = Object.entries(timeframes).map(([tf, data]) => {
                const score = data.score;
                const arrow = score > 0.1 ? '▲' : score < -0.1 ? '▼' : '●';
                const arrowClass = score > 0.1 ? 'up' : score < -0.1 ? 'down' : 'neutral';

                return `
                    <div class="tf-signal">
                        <span class="tf-name">${tf}</span>
                        <span class="arrow ${arrowClass}">${arrow}</span>
                    </div>
                `;
            }).join('');
        }
    }

    updateTradeDisplay(suggestion) {
        const card = document.getElementById('trade-card');
        const directionEl = document.getElementById('trade-direction');

        if (!suggestion) {
            directionEl.className = 'trade-direction';
            directionEl.innerHTML = `
                <span class="direction-icon">—</span>
                <span class="direction-text">No Signal</span>
            `;
            return;
        }

        const isLong = suggestion.direction === 'LONG';

        directionEl.className = 'trade-direction ' + (isLong ? 'long' : 'short');
        directionEl.innerHTML = `
            <span class="direction-icon">${isLong ? '📈' : '📉'}</span>
            <span class="direction-text">${suggestion.direction}</span>
        `;

        // Update levels
        document.getElementById('entry-price').textContent = `$${suggestion.entry.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        document.getElementById('stop-loss').textContent = `$${suggestion.stop_loss.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        document.getElementById('tp1').textContent = `$${suggestion.take_profit_1.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        document.getElementById('tp2').textContent = `$${suggestion.take_profit_2.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

        // Meta
        document.getElementById('rr-ratio').textContent = `1:${suggestion.risk_reward.toFixed(1)}`;
        document.getElementById('confidence').textContent = `${Math.round(suggestion.confidence * 100)}%`;
    }

    updateHistoryDisplay(history) {
        const container = document.getElementById('history-list');
        if (!container) return;

        if (!history || history.length === 0) {
            container.innerHTML = '<div class="no-history">No signals yet</div>';
            return;
        }

        container.innerHTML = history.slice().reverse().slice(0, 10).map(item => {
            const time = new Date(item.timestamp).toLocaleTimeString();
            const isLong = item.direction === 'LONG';

            return `
                <div class="history-item">
                    <span class="history-time">${time}</span>
                    <span class="history-direction ${isLong ? 'long' : 'short'}">${item.direction}</span>
                    <span class="history-price">$${item.price.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
                </div>
            `;
        }).join('');
    }

    // ==================== Alerts ====================

    showAlert(data) {
        const modal = document.getElementById('alert-modal');
        const content = modal.querySelector('.alert-content');
        const icon = document.getElementById('alert-icon');
        const title = document.getElementById('alert-title');
        const price = document.getElementById('alert-price');
        const score = document.getElementById('alert-score');

        const isLong = data.suggestion.direction === 'LONG';

        content.className = 'alert-content ' + (isLong ? 'bullish' : 'bearish');
        icon.textContent = isLong ? '📈' : '📉';
        title.textContent = `${isLong ? 'BULLISH' : 'BEARISH'} SIGNAL`;
        title.className = 'alert-title ' + (isLong ? 'bullish' : 'bearish');
        price.textContent = `$${data.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        score.textContent = `Score: ${data.signal.combined_score >= 0 ? '+' : ''}${data.signal.combined_score.toFixed(2)}`;

        modal.classList.remove('hidden');

        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            modal.classList.add('hidden');
        }, 10000);
    }

    playSound(direction) {
        if (!this.soundEnabled) return;

        const soundId = direction === 'LONG' ? 'bullish-sound' : 'bearish-sound';
        const audio = document.getElementById(soundId);

        if (audio) {
            audio.currentTime = 0;
            audio.play().catch(e => console.log('Audio play failed:', e));
        }
    }

    // ==================== Event Listeners ====================

    setupEventListeners() {
        // Timeframe buttons
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentTimeframe = btn.dataset.tf;
                this.saveToSession('currentTimeframe', this.currentTimeframe);
                // Notify backend of timeframe change so it only fetches this timeframe
                this.socket.emit('set_timeframe', { timeframe: this.currentTimeframe });
                // Reset all flags when changing timeframes so chart fits content
                this.userHasInteracted = false;
                this.isInitialLoad = true;
                this.hasLoadedData = false;
                this.loadChartData(this.currentTimeframe, false);
            });
        });

        // Set active timeframe button from session on load
        const activeTimeframeBtn = document.querySelector(`.tf-btn[data-tf="${this.currentTimeframe}"]`);
        if (activeTimeframeBtn) {
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            activeTimeframeBtn.classList.add('active');
        }

        // Indicator tabs
        document.querySelectorAll('.ind-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.ind-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.switchIndicator(tab.dataset.indicator);
            });
        });

        // Set active indicator tab from session on load
        const activeIndicatorTab = document.querySelector(`.ind-tab[data-indicator="${this.currentIndicator}"]`);
        if (activeIndicatorTab) {
            document.querySelectorAll('.ind-tab').forEach(t => t.classList.remove('active'));
            activeIndicatorTab.classList.add('active');
        }

        // Sound toggle
        const soundToggle = document.getElementById('sound-toggle');
        if (soundToggle) {
            // Set initial state from session
            soundToggle.querySelector('.sound-on').classList.toggle('hidden', !this.soundEnabled);
            soundToggle.querySelector('.sound-off').classList.toggle('hidden', this.soundEnabled);

            soundToggle.addEventListener('click', () => {
                this.soundEnabled = !this.soundEnabled;
                this.saveToSession('soundEnabled', this.soundEnabled);
                soundToggle.querySelector('.sound-on').classList.toggle('hidden', !this.soundEnabled);
                soundToggle.querySelector('.sound-off').classList.toggle('hidden', this.soundEnabled);
            });
        }

        // Alert close
        const alertClose = document.getElementById('alert-close');
        const alertModal = document.getElementById('alert-modal');
        if (alertClose && alertModal) {
            alertClose.addEventListener('click', () => {
                alertModal.classList.add('hidden');
            });

            alertModal.addEventListener('click', (e) => {
                if (e.target === alertModal) {
                    alertModal.classList.add('hidden');
                }
            });
        }
    }

    switchIndicator(indicator) {
        this.currentIndicator = indicator;
        this.saveToSession('currentIndicator', indicator);

        // Hide all
        Object.values(this.indicatorSeries).forEach(series => {
            series.applyOptions({ visible: false });
        });

        // Show selected
        switch (indicator) {
            case 'macd':
                this.indicatorSeries.macd_line.applyOptions({ visible: true });
                this.indicatorSeries.macd_signal.applyOptions({ visible: true });
                this.indicatorSeries.macd_histogram.applyOptions({ visible: true });
                break;
            case 'rsi':
                this.indicatorSeries.rsi.applyOptions({ visible: true });
                break;
            case 'volume':
                this.indicatorSeries.volume.applyOptions({ visible: true });
                break;
        }
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.analyzer = new BTCAnalyzer();
});
