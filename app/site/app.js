const SCALE = [[0, "#2c7fb8"], [0.5, "#fec44f"], [1, "#e34a33"]];
let DATA = null, sel = -1;

fetch("data.json").then(r => r.json()).then(d => {
  DATA = d;
  drawScatter();
  // default selection: a high-surprise organoid (most "surprising")
  select(DATA.organoids.reduce((bi, o, i, a) => o.surprise > a[bi].surprise ? i : bi, 0));
});

function drawScatter() {
  const o = DATA.organoids;
  const pts = {
    x: o.map(d => d.mean_speed), y: o.map(d => d.surprise),
    customdata: o.map((d, i) => i),
    text: o.map(d => `${d.batch}<br>surprise ${d.surprise.toExponential(2)}<br>${d.percentile}th pct`),
    mode: "markers", type: "scattergl", hoverinfo: "text",
    marker: {
      size: 11, color: o.map(d => d.percentile), colorscale: SCALE, cmin: 0, cmax: 100,
      line: { color: "#0f1115", width: 1 },
      colorbar: { title: "percentile", thickness: 10, len: 0.6, tickfont: { color: "#9aa3b2" } }
    }
  };
  const hl = { x: [], y: [], mode: "markers", type: "scattergl", hoverinfo: "skip",
    marker: { size: 20, color: "rgba(0,0,0,0)", line: { color: "#fff", width: 2.5 } } };
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9aa3b2", size: 12 }, showlegend: false,
    margin: { l: 60, r: 10, t: 10, b: 45 },
    xaxis: { title: "mean speed (px/frame)", gridcolor: "#262b36", zeroline: false },
    yaxis: { title: "prediction surprise", type: "log", gridcolor: "#262b36", zeroline: false }
  };
  Plotly.newPlot("scatter", [pts, hl], layout, { displayModeBar: false, responsive: true });
  document.getElementById("scatter").on("plotly_click", e => {
    if (e.points[0].data.mode === "markers" && e.points[0].customdata !== undefined)
      select(e.points[0].customdata);
  });
}

function select(i) {
  sel = i;
  const d = DATA.organoids[i], pct = d.percentile, n = DATA.n;
  document.getElementById("vid").src = d.clip;
  document.getElementById("org-title").textContent = d.batch;
  const harder = `harder to predict than ${pct}% of the population`;
  const easier = `easier to predict than ${100 - pct}% of the population`;
  document.getElementById("readout").innerHTML =
    `In the <span class="pct">${pct}th percentile</span> of prediction surprise — the ` +
    `population-trained model finds this organoid's motion ${pct >= 50 ? harder : easier}.`;
  document.getElementById("metrics").innerHTML = [
    ["Prediction surprise Sₒ", d.surprise.toExponential(2)],
    ["Surprise rank", `${d.rank} / ${n}`],
    ["Mean speed", d.mean_speed != null ? d.mean_speed.toFixed(2) : "—"],
    ["Size (px)", d.size_px != null ? Math.round(d.size_px) : "—"],
  ].map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  Plotly.restyle("scatter", { x: [[d.mean_speed]], y: [[d.surprise]] }, [1]);  // highlight ring
}
