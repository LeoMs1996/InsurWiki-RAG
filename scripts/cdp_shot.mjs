// Minimal CDP driver: navigate, evaluate JS (interact), wait, screenshot to file.
// Usage: node cdp_shot.mjs --url URL [--eval JS]... [--wait ms] [--shot out.png] [--full] [--timeout ms]
import fs from 'node:fs';

const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const url = get('url');
const evals = args.filter((a, i) => args[i - 1] === '--eval');
const wait = parseInt(get('wait', '500'), 10);
const shot = get('shot');
const full = args.includes('--full');
const timeout = parseInt(get('timeout', '20000'), 10);

const BASE = 'http://127.0.0.1:9223';
async function findTarget() {
  const r = await fetch(BASE + '/json/list');
  const ts = await r.json();
  return ts.find(t => t.type === 'page');
}
const t = await findTarget();
if (!t) { console.error('NO_PAGE_TARGET'); process.exit(2); }

const ws = new WebSocket(t.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let id = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const mid = ++id;
    pending.set(mid, { resolve, reject });
    ws.send(JSON.stringify({ id: mid, method, params }));
    setTimeout(() => { if (pending.has(mid)) { pending.delete(mid); reject(new Error('TIMEOUT ' + method)); } }, timeout);
  });
}
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    const p = pending.get(m.id); pending.delete(m.id);
    if (m.error) p.reject(new Error(m.error.message)); else p.resolve(m.result);
  } else if (m.method === 'Runtime.exceptionThrown') {
    console.error('PAGE_EXCEPTION', JSON.stringify(m.params.exceptionDetails).slice(0, 500));
  }
};

async function evalJS(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error('EVAL_ERROR: ' + JSON.stringify(r.exceptionDetails).slice(0, 500));
  return r.result?.value;
}

await send('Page.enable');
await send('Runtime.enable');

if (url) {
  const nav = send('Page.navigate', { url }).then(r => {
    if (r.errorText) throw new Error('NAV_ERROR ' + r.errorText);
  });
  // wait for load event
  await new Promise(async (resolve) => {
    const timer = setTimeout(resolve, timeout);
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.id && pending.has(m.id)) {
        const p = pending.get(m.id); pending.delete(m.id);
        if (m.error) p.reject(new Error(m.error.message)); else p.resolve(m.result);
      }
      if (m.method === 'Page.loadEventFired') { clearTimeout(timer); setTimeout(resolve, 400); }
    };
    await nav;
  });
}

let lastResult;
for (const js of evals) lastResult = await evalJS(js);
if (evals.length && lastResult !== undefined) console.log('EVAL_RESULT:', JSON.stringify(lastResult)?.slice(0, 300));

if (wait > 0) await new Promise(r => setTimeout(r, wait));

if (shot) {
  let params = { format: 'png' };
  if (full) {
    const metrics = await send('Page.getLayoutMetrics');
    const { width, height } = metrics.cssContentSize || metrics.contentSize;
    await send('Emulation.setDeviceMetricsOverride', { width: Math.ceil(width), height: Math.ceil(height), deviceScaleFactor: 1, mobile: false });
    params.clip = { x: 0, y: 0, width: Math.ceil(width), height: Math.ceil(height), scale: 1 };
  }
  await new Promise(r => setTimeout(r, 300));
  const r = await send('Page.captureScreenshot', params);
  fs.writeFileSync(shot, Buffer.from(r.data, 'base64'));
  console.log('SAVED:', shot, fs.statSync(shot).size, 'bytes');
  if (full) await send('Emulation.clearDeviceMetricsOverride');
}
ws.close();
process.exit(0);
