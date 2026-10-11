/* Display precomputed, training-only output-map fits for whole-block subsets. */
(() => {
  'use strict';
  const data = window.lrnnReductionData;
  if (!data) return;
  const get = (id) => document.getElementById(id);
  const choices = data.blocks.map((_, i) => get(`retain-block-${i}`));
  const refit = get('refit-map');
  const ns = 'http://www.w3.org/2000/svg';
  const make = (tag, attributes, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const format = (value) => value < 1e-3 ? value.toExponential(2) : value.toFixed(4);
  const fullMagnitude = Math.max(...data.fullMatrix.flat().map(Math.abs), 1);
  const signalLimit = Math.ceil(Math.max(
    ...data.target.map(Math.abs),
    ...data.subsets.flatMap(s => [...s.prediction, ...s.fixedPrediction]).map(Math.abs)
  ) * 2) / 2;

  function matrix(id, values, selectedMask = 15, showOmission = false) {
    const svg = get(id);
    svg.replaceChildren();
    const n = values.length;
    if (!n) {
      svg.append(make('text', {x:90,y:88,'text-anchor':'middle',class:'matrix-empty'}, 'No retained blocks'));
      return;
    }
    const cell = 18;
    const offset = (180 - n * cell) / 2;
    values.forEach((row, r) => row.forEach((value, c) => {
      const removed = showOmission && Math.floor(r / 2) === Math.floor(c / 2)
        && !(selectedMask & (1 << Math.floor(r / 2)));
      const fill = Math.abs(value) < 1e-14 ? '#edf1f7' : (removed ? '#c7d0dc' : '#1859a6');
      const opacity = Math.abs(value) < 1e-14 || removed ? 1 : 0.2 + 0.8 * Math.abs(value) / fullMagnitude;
      svg.append(make('rect', {x:offset + c*cell, y:offset + r*cell,width:cell-2,height:cell-2,rx:1,fill,opacity}));
    }));
    if (showOmission) {
      for (let i = 0; i < data.blocks.length; i++) {
        const group = make('g', {role:'button',tabindex:0,'data-block':i,'aria-label':`Toggle block ${i+1}`,'aria-pressed':Boolean(selectedMask & (1 << i))});
        group.append(make('rect', {x:offset+i*2*cell-2,y:offset+i*2*cell-2,width:cell*2,height:cell*2,rx:3,
          fill:'transparent',stroke:selectedMask & (1<<i) ? '#1859a6':'#9aa7b7','stroke-width':1.5}));
        const toggle = () => {
          choices[i].checked = !choices[i].checked;
          update();
          get('blocks-matrix').querySelector(`[data-block="${i}"]`).focus();
        };
        group.addEventListener('click', toggle);
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
        });
        svg.append(group);
      }
    }
  }

  function plot(id, values, residual = false) {
    const svg = get(id);
    svg.replaceChildren();
    const x0 = 56, width = 864, top = 32, bottom = residual ? 119 : 214;
    const limit = residual ? Math.max(0.05, Math.ceil(Math.max(...values.map(Math.abs)) * 20) / 20) : signalLimit;
    const x = (t) => x0 + width * t / (data.length - 1);
    const y = (v) => (top + bottom) / 2 - v * (bottom - top) / (2*limit);
    const split = x(data.trainLength - 0.5);
    svg.append(make('rect', {x:x0,y:top,width:split-x0,height:bottom-top,fill:'#f2f5fa'}));
    svg.append(make('rect', {x:split,y:top,width:x(data.length-1)-split,height:bottom-top,fill:'#f8f5ef'}));
    const ticks = residual ? [-limit, 0, limit] : [-limit, -limit/2, 0, limit/2, limit];
    for (const value of ticks) {
      svg.append(make('line',{x1:x0,y1:y(value),x2:x(data.length-1),y2:y(value),stroke:'#dce3ec','stroke-width':1}));
      svg.append(make('text',{x:46,y:y(value)+4,'text-anchor':'end',class:'plot-tick'},value === 0 ? '0' : String(Number(value.toFixed(2)))));
    }
    svg.append(make('line',{x1:split,y1:top,x2:split,y2:bottom,stroke:'#b1bdcb','stroke-dasharray':'4 4'}));
    const path = (series) => series.map((v,t)=>`${t?'L':'M'}${x(t).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
    if (!residual) {
      svg.append(make('text',{x:205,y:18,'text-anchor':'middle',class:'plot-label'},'Training · 120 samples'));
      svg.append(make('text',{x:743,y:18,'text-anchor':'middle',class:'plot-label'},'Later samples · 120'));
      svg.append(make('path',{d:path(data.target),fill:'none',stroke:'#7a8595','stroke-width':2}));
    } else {
      svg.append(make('text',{x:x0,y:18,class:'plot-label'},'Residual: approximation − original signal'));
    }
    svg.append(make('path',{d:path(values),fill:'none',stroke:residual?'#ab5b1a':'#1859a6','stroke-width':2.2,
      ...(residual ? {} : {'stroke-dasharray':'6 3'})}));
    for (const t of [0,60,120,180,239]) {
      svg.append(make('text',{x:x(t),y:bottom+20,'text-anchor':'middle',class:'plot-tick'},t));
    }
    if (!residual) {
      svg.append(make('text',{x:width/2+x0,y:253,'text-anchor':'middle',class:'plot-label'},'Time step'));
    }
  }

  function update() {
    const mask = choices.reduce((m, checkbox, i) => m | (checkbox.checked ? (1 << i) : 0), 0);
    const subset = data.subsets[mask];
    const prediction = refit.checked ? subset.prediction : subset.fixedPrediction;
    const residual = prediction.map((v,t)=>v-data.target[t]);
    matrix('blocks-matrix', data.jordanMatrix, mask, true);
    matrix('retained-matrix', subset.matrix);
    get('retained-dimension').textContent = `${subset.dimension} state values`;
    get('retained-matrix').setAttribute('aria-label',`Retained transition matrix with ${subset.dimension} state values`);
    get('reduction-size').textContent = `${subset.dimension} / 8`;
    get('reduction-train').textContent = format(refit.checked ? subset.trainingRmse : subset.fixedTrainingRmse);
    get('reduction-future').textContent = format(refit.checked ? subset.futureRmse : subset.fixedFutureRmse);
    choices.forEach((checkbox,i)=>checkbox.closest('label').classList.toggle('block-omitted',!checkbox.checked));
    get('remove-block').disabled = mask === 0;
    plot('reduction-signal', prediction);
    plot('reduction-residual', residual, true);
    get('reduction-fit-description').textContent = mask === 0
      ? 'With no retained blocks, the approximation is zero. RMSE is the square root of the mean squared residual over each interval.'
      : `${refit.checked ? 'The output map is refitted using steps 0–119 only.' : 'The original output coefficients are retained without refitting.'} Grey: original signal. Dashed blue: approximation. RMSE is the square root of the mean squared residual over each interval. The signal axis is fixed; the labelled residual axis adjusts to the selected error.`;
  }

  data.blocks.forEach((block,i)=>{
    get(`block-score-${i}`).textContent = `Omission RMSE: ${format(block.omissionRmse)}`;
  });
  choices.forEach(checkbox=>checkbox.addEventListener('change',update));
  refit.addEventListener('change',update);
  get('remove-block').addEventListener('click',()=>{
    const index = [...data.ranking].reverse().find(i=>choices[i].checked);
    if (index !== undefined) { choices[index].checked=false; update(); }
  });
  get('restore-blocks').addEventListener('click',()=>{choices.forEach(c=>{c.checked=true;});update();});
  matrix('full-matrix', data.fullMatrix);
  update();
  get('reduction-interactive').hidden = false;
  get('reduction-fallback').hidden = true;
})();
