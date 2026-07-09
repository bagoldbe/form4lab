// Chart initialization using Lightweight Charts
//
// Colors reference CSS custom properties defined in style.css.
// We read them once via getComputedStyle so charts match the theme.

var _rootStyle = getComputedStyle(document.documentElement);
function _css(prop) { return _rootStyle.getPropertyValue(prop).trim(); }

var CHART_COLORS = {
  bg: _css('--bg-primary') || '#0d1117',
  text: _css('--text-primary') || '#c9d1d9',
  grid: _css('--bg-tertiary') || '#21262d',
  border: _css('--border-color') || '#30363d',
  up: _css('--positive-color') || '#3fb950',
  down: _css('--negative-color') || '#f85149',
  link: _css('--link-color') || '#58a6ff',
};


function destroyChart(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return;
  if (container._resizeHandler) {
    window.removeEventListener('resize', container._resizeHandler);
    container._resizeHandler = null;
  }
  if (container._chart) {
    container._chart.remove();
    container._chart = null;
  }
}


function initChart(containerId, priceData, markers) {
  if (!window.LightweightCharts) return;
  var container = document.getElementById(containerId);
  if (!container) return;

  // Clean up any existing chart on this container
  destroyChart(containerId);

  var chart = LightweightCharts.createChart(container, {
    layout: {
      background: { color: CHART_COLORS.bg },
      textColor: CHART_COLORS.text,
    },
    grid: {
      vertLines: { color: CHART_COLORS.grid },
      horzLines: { color: CHART_COLORS.grid },
    },
    width: container.clientWidth,
    height: 400,
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
      borderColor: CHART_COLORS.border,
    },
    timeScale: {
      borderColor: CHART_COLORS.border,
      timeVisible: false,
    },
  });

  var candleSeries = chart.addCandlestickSeries({
    upColor: CHART_COLORS.up,
    downColor: CHART_COLORS.down,
    borderUpColor: CHART_COLORS.up,
    borderDownColor: CHART_COLORS.down,
    wickUpColor: CHART_COLORS.up,
    wickDownColor: CHART_COLORS.down,
  });

  candleSeries.setData(priceData);

  if (markers && markers.length > 0) {
    markers.sort(function(a, b) {
      return a.time < b.time ? -1 : a.time > b.time ? 1 : 0;
    });
    candleSeries.setMarkers(markers);
  }

  chart.timeScale().fitContent();

  var resizeHandler = function() {
    chart.applyOptions({ width: container.clientWidth });
  };
  window.addEventListener('resize', resizeHandler);

  container._chart = chart;
  container._resizeHandler = resizeHandler;

  return chart;
}


function initVolumeChart(containerId, priceData) {
  if (!window.LightweightCharts) return;
  var container = document.getElementById(containerId);
  if (!container) return;

  destroyChart(containerId);

  var chart = LightweightCharts.createChart(container, {
    layout: {
      background: { color: CHART_COLORS.bg },
      textColor: CHART_COLORS.text,
    },
    grid: {
      vertLines: { color: CHART_COLORS.grid },
      horzLines: { color: CHART_COLORS.grid },
    },
    width: container.clientWidth,
    height: 150,
    rightPriceScale: {
      borderColor: CHART_COLORS.border,
    },
    timeScale: {
      borderColor: CHART_COLORS.border,
      visible: false,
    },
  });

  var volumeSeries = chart.addHistogramSeries({
    color: CHART_COLORS.link,
    priceFormat: {
      type: 'volume',
    },
  });

  var volumeData = priceData.map(function(d) {
    return {
      time: d.time,
      value: d.volume || 0,
      color: d.close >= d.open ? CHART_COLORS.up + '60' : CHART_COLORS.down + '60',
    };
  });

  volumeSeries.setData(volumeData);
  chart.timeScale().fitContent();

  var resizeHandler = function() {
    chart.applyOptions({ width: container.clientWidth });
  };
  window.addEventListener('resize', resizeHandler);

  container._chart = chart;
  container._resizeHandler = resizeHandler;

  return chart;
}


// Two-line chart for portfolio equity vs SPY benchmark, with event markers.
// payload shape: { dates:[], equity:[], benchmark:[], events:[{date,label,color}], summary:{} }
function initEquityChart(containerId, payload) {
  if (!window.LightweightCharts) return;
  var container = document.getElementById(containerId);
  if (!container) return;

  destroyChart(containerId);

  if (!payload || !payload.dates || payload.dates.length === 0) {
    container.innerHTML = '<div style="padding:1rem; color: var(--text-secondary); text-align:center;">No equity history yet.</div>';
    return;
  }

  var chart = LightweightCharts.createChart(container, {
    layout: {
      background: { color: CHART_COLORS.bg },
      textColor: CHART_COLORS.text,
    },
    grid: {
      vertLines: { color: CHART_COLORS.grid },
      horzLines: { color: CHART_COLORS.grid },
    },
    width: container.clientWidth,
    height: 300,
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: CHART_COLORS.border },
    timeScale: { borderColor: CHART_COLORS.border, timeVisible: false },
  });

  var equitySeries = chart.addLineSeries({
    color: CHART_COLORS.link,
    lineWidth: 2,
    title: 'Portfolio',
  });
  var benchSeries = chart.addLineSeries({
    color: CHART_COLORS.text,
    lineWidth: 1,
    lineStyle: 2,
    title: 'SPY benchmark',
  });

  var equityData = payload.dates.map(function(d, i) {
    return { time: d, value: payload.equity[i] };
  });
  var benchData = payload.dates.map(function(d, i) {
    return { time: d, value: payload.benchmark[i] };
  });
  equitySeries.setData(equityData);
  benchSeries.setData(benchData);

  if (payload.events && payload.events.length > 0) {
    var markers = payload.events.map(function(e) {
      return {
        time: e.date,
        position: 'inBar',
        color: e.color,
        shape: 'circle',
        text: e.label,
      };
    });
    markers.sort(function(a, b) { return a.time < b.time ? -1 : a.time > b.time ? 1 : 0; });
    equitySeries.setMarkers(markers);
  }

  chart.timeScale().fitContent();

  var resizeHandler = function() {
    chart.applyOptions({ width: container.clientWidth });
  };
  window.addEventListener('resize', resizeHandler);

  container._chart = chart;
  container._resizeHandler = resizeHandler;
  return chart;
}
