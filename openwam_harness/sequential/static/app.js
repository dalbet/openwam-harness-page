"use strict";
const $ = (id) => document.getElementById(id);
const suiteSel = $("suite"), taskSel = $("task-id"), seedIn = $("seed"),
  loadBtn = $("load"), taskIn = $("task"),
  startBtn = $("start"), manualBtn = $("manual"), longBtn = $("long"),
  restartBtn = $("restart"), stopBtn = $("stop"),
  okBtn = $("ok"), badBtn = $("bad"), unkBtn = $("unk"),
  agentCanvas = $("agent"), wristCanvas = $("wrist"),
  agentView = $("agent-view"), wristView = $("wrist-view"),
  phAgent = $("ph-agent"), phWrist = $("ph-wrist"), statusEl = $("status"),
  healthEl = $("health"), badge = $("badge"),
  plannerMeta = $("planner-meta"), plannerBrief = $("planner-brief"), plannerFull = $("planner-full");

const agCtx = agentCanvas.getContext("2d");
const wrCtx = wristCanvas.getContext("2d");

let pending = null;
let decoding = false;
let curEid = null;
let acceptEid = null;
let gate = false;
let lastDisplayedFid = -1;
let terminalReached = false;
let lastArrival = 0;
let endedFallback = null;
let episode = null;
let sceneEpoch = null;
let planVersion = 1;
let waiting = false;

function clearBuffer(){
  pending = null; decoding = false; lastDisplayedFid = -1; endedFallback = null;
}
function setStatus(phase, detail, cls){
  const c = cls ? ` ${cls}` : "";
  statusEl.innerHTML = `<span class="phase${c}">${escapeHtml(phase)}</span>` + (detail ? ` — ${escapeHtml(detail)}` : "");
}
function escapeHtml(value){
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[ch]));
}
function setConfirm(on){
  waiting = on;
  okBtn.disabled = !on;
  badBtn.disabled = !on;
  unkBtn.disabled = !on;
  stopBtn.disabled = !on && startBtn.disabled;
}

const PHASES = {
  queued: "排队",
  processing_input: "处理输入",
  generating: "生成中",
  validating: "校验",
  planning: "planning",
  loading: "加载中",
  loaded: "场景已加载",
  executing: "executing",
  waiting_confirmation: "waiting_confirmation",
  failed: "failed"
};
let suites = [];
let ws = null, wsConnected = false;
let lastHealth = null;
let suitePoll = null;
function connect(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsConnected = true; setStatus("就绪", location.host); send({type:"get_scenes"}); send({type:"get_health"}); };
  ws.onclose = () => {
    wsConnected = false;
    setStatus("断开", "与服务器的连接已断开，尝试重连…", "warn");
    setConfirm(false);
    clearBuffer();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => { try{ ws.close(); }catch(e){} };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleText(ev.data);
    else handleBinary(ev.data);
  };
}
function send(obj){ if(ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function handleText(txt){
  let m; try{ m = JSON.parse(txt); }catch(e){ return; }
  switch(m.type){
    case "health":
      renderHealth(m);
      break;
    case "suites":
      if (m.status === "ready") applyCatalog(m.suites || []);
      break;
    case "loaded":
      curEid = null; acceptEid = null; gate = false; clearBuffer();
      episode = m.episode_id; sceneEpoch = m.scene_epoch; planVersion = m.plan_version || 1;
      taskIn.value = m.instruction || "";
      $("summary").textContent = m.summary_zh ? `中文摘要：${m.summary_zh}` : "";
      taskIn.disabled = false; manualBtn.disabled = false;
      startBtn.disabled = !(lastHealth && lastHealth.can_model_plan);
      longBtn.disabled = !currentTask().long_candidate;
      restartBtn.disabled = true; loadBtn.disabled = false;
      setConfirm(false);
      if (m.placeholder || m.mode === "dry"){
        setStatus("干跑", "未创建真实环境，没有新图像", "bad");
        phAgent.textContent = "干跑，无图像"; phWrist.textContent = "干跑，无图像";
      } else {
        const scene = m.scene || {};
        setStatus("场景已加载", `${m.env_id} · step ${scene.env_step ?? "?"} · ${ (scene.object_names||[]).join(", ") }`, "good");
        phAgent.textContent = "等待画面"; phWrist.textContent = "等待画面";
      }
      break;
    case "started":
      acceptEid = m.episode_id; gate = false; curEid = m.episode_id;
      episode = m.episode_id;
      clearBuffer(); terminalReached = false;
      stopBtn.disabled = false;
      setStatus(m.mode === "dry" ? "干跑" : "planning", m.plan_source || "");
      break;
    case "load_failed":
      loadBtn.disabled = false; startBtn.disabled = false;
      setStatus("failed", m.detail || "加载失败", "bad");
      break;
    case "state":
      if (m.scene_epoch != null) sceneEpoch = m.scene_epoch;
      if (m.plan_version != null) planVersion = m.plan_version;
      if (m.phase === "waiting_confirmation") setConfirm(true);
      else if (m.phase === "failed" || m.phase === "executing" || m.phase === "planning") setConfirm(false);
      if (!terminalReached) setStatus(PHASES[m.phase] || m.phase || "state", statusDetail(m), m.phase === "failed" ? "bad" : "");
      const verify = m.verification_status ? ` · 验证 ${m.verification_status}` : "";
      badge.textContent = `plan_source=${m.plan_source || "—"} · 恢复 ${m.recovery_attempts || 0} 次${verify}`;
      break;
    case "planner_io":
      renderPlanner(m);
      break;
    case "log":
      $("log").textContent += (m.text || "") + "\n";
      break;
    case "ended":
      if (acceptEid && m.episode_id && m.episode_id !== acceptEid) break;
      endedFallback = m;
      setConfirm(false);
      loadBtn.disabled = false; startBtn.disabled = false; manualBtn.disabled = false; restartBtn.disabled = false; stopBtn.disabled = true;
      break;
    case "pong": break;
  }
}

function statusDetail(m){
  const timing = m.timings ? JSON.stringify(m.timings) : "";
  const bits = [m.detail || "", m.reason || "", m.reason_code || "", m.step_id || "", m.skill || "", m.evidence_source || "", timing].filter(Boolean);
  return bits.join(" · ");
}
function currentTask(){
  const suite = suites.find(item => item.id === suiteSel.value);
  const task = suite && (suite.tasks || []).find(item => String(item.id) === taskSel.value);
  return task || {};
}
function applyCatalog(items){
  suites = items || [];
  if (!suites.length) return;
  fillSuites();
  suiteSel.disabled = false;
  taskSel.disabled = false;
  seedIn.disabled = false;
  loadBtn.disabled = false;
  if (suitePoll) { clearInterval(suitePoll); suitePoll = null; }
}
function fillSuites(){
  suiteSel.innerHTML = "";
  suites.forEach(suite => {
    const option = document.createElement("option");
    option.value = suite.id;
    option.textContent = `${suite.label} / ${suite.id}`;
    suiteSel.appendChild(option);
  });
  fillTasks();
}
function fillTasks(){
  const suite = suites.find(item => item.id === suiteSel.value);
  taskSel.innerHTML = "";
  (suite && suite.tasks || []).forEach(task => {
    const option = document.createElement("option");
    option.value = String(task.id);
    option.textContent = `t${task.id} ${task.language}`;
    taskSel.appendChild(option);
  });
  applyTask();
}
function applyTask(){
  const task = currentTask();
  taskIn.value = task.language || "";
  $("summary").textContent = task.summary_zh ? `中文摘要：${task.summary_zh}` : "";
  const maxSeed = Math.max(0, (task.init_states || 1) - 1);
  seedIn.max = String(maxSeed);
  if (Number(seedIn.value) > maxSeed) seedIn.value = "0";
  longBtn.disabled = !task.long_candidate || !episode;
}

function renderHealth(m){
  lastHealth = m;
  const mode = m.mode === "live" ? "真实模式" : "干跑模式";
  const planner = m.planner === "ok" ? "planner 正常" : `planner ${m.planner || "未知"}`;
  const env = {ready: "env 就绪", loading: "env 加载中", down: "env 已停止", not_loaded: "env 未加载"}[m.env] || `env ${m.env || "未知"}`;
  const vla = {ready: "VLA 就绪", down: "VLA 已停止", not_loaded: "VLA 未加载"}[m.vla] || `VLA ${m.vla || "未知"}`;
  healthEl.textContent = `适配器在线 · ${mode} · ${location.host} · pid ${m.pid} · ${planner} · ${env} · ${vla}`;
  healthEl.style.color = m.mode === "live" ? "var(--dim)" : "var(--warn)";
  if (m.error || m.planner_detail) healthEl.title = [m.error, m.planner_detail].filter(Boolean).join("\n");
  $("retry").hidden = true;
  if (m.can_model_plan === false && !startBtn.disabled && taskIn.value.trim()) startBtn.disabled = true;
}

function renderPlanner(m){
  const usage = m.usage || {};
  const tokens = `prompt ${usage.prompt_tokens ?? "—"} / completion ${usage.completion_tokens ?? "—"} / total ${usage.total_tokens ?? "—"}`;
  const source = m.plan_source === "manual" ? "人工计划" : (m.plan_source === "model" ? "模型" : (m.plan_source || "无计划"));
  plannerMeta.textContent = `${source} · ${m.model || ""} · ${Number(m.elapsed_s || 0).toFixed(1)}s · ${tokens} · 修复 ${m.repairs || 0} 次 · ${m.response_kind || ""}`;
  const errors = (m.errors || []).map(item => `${item.path || ""} ${item.code || ""} received=${JSON.stringify(item.received)} expected ${item.expected || ""}`).join("\n");
  plannerBrief.textContent = [
    "输入摘要: " + (m.summary || ""),
    "图像: " + ((m.image_ids || []).join(", ") || "无"),
    "解析: " + (errors ? "失败" : "通过"),
    errors ? "错误: " + errors : "",
    "输出摘要: " + String(m.output || "").slice(0, 280)
  ].filter(Boolean).join("\n");
  plannerFull.hidden = true;
  plannerFull.textContent = String(m.output || "");
  badge.textContent = `plan_source=${m.plan_source || "—"}`;
}

function handleBinary(ab){
  const u8 = new Uint8Array(ab);
  const dv = new DataView(ab);
  const hlen = dv.getUint32(0);
  let hdr; try{ hdr = JSON.parse(new TextDecoder().decode(u8.subarray(4, 4+hlen))); }catch(e){ return; }
  if (gate) return;
  if (acceptEid !== null && hdr.eid && hdr.eid !== acceptEid) return;
  let off = 4 + hlen;
  const aLen = hdr.a_len|0, wLen = hdr.w_len|0;
  const frame = {
    hdr,
    aBlob: u8.subarray(off, off+aLen),
    wBlob: u8.subarray(off+aLen, off+aLen+wLen),
    arrival: performance.now(),
    gap: lastArrival ? performance.now() - lastArrival : 0
  };
  lastArrival = frame.arrival;
  if (curEid !== null && hdr.eid && hdr.eid !== curEid) clearBuffer();
  curEid = hdr.eid || curEid;
  if (!hdr.term) terminalReached = false;
  if (hdr.term) pending = frame;
  else if (!pending || !pending.hdr.term) pending = frame;
}

function showResult(success, reason){
  terminalReached = true;
  setStatus(success ? "完成" : "failed", reason || "", success ? "good" : "bad");
  startBtn.disabled = false; manualBtn.disabled = false; restartBtn.disabled = false; loadBtn.disabled = false;
  setConfirm(false);
  endedFallback = null;
}

function applyAspect(view, width, height){
  if (width > 0 && height > 0) view.style.setProperty("--ar", `${width} / ${height}`);
}

function tick(){
  requestAnimationFrame(tick);
  if (endedFallback && !pending && !decoding && performance.now() - (endedFallback._at || (endedFallback._at = performance.now())) > 800){
    showResult(!!endedFallback.success, endedFallback.reason);
    return;
  }
  if (decoding || !pending) return;
  const f = pending;
  pending = null;
  decoding = true;
  Promise.all([
    createImageBitmap(new Blob([f.aBlob], {type:"image/jpeg"})),
    createImageBitmap(new Blob([f.wBlob], {type:"image/jpeg"}))
  ]).then(([aBmp, wBmp]) => {
    agentCanvas.width = aBmp.width; agentCanvas.height = aBmp.height;
    wristCanvas.width = wBmp.width; wristCanvas.height = wBmp.height;
    applyAspect(agentView, f.hdr.aw || aBmp.width, f.hdr.ah || aBmp.height);
    applyAspect(wristView, f.hdr.ww || wBmp.width, f.hdr.wh || wBmp.height);
    agCtx.drawImage(aBmp, 0, 0);
    wrCtx.drawImage(wBmp, 0, 0);
    aBmp.close(); wBmp.close();
    phAgent.style.display = "none"; phWrist.style.display = "none";
    lastDisplayedFid = f.hdr.fid;
    if (f.hdr.phase === "waiting_confirmation") setConfirm(true);
    if (f.hdr.term) showResult(!!f.hdr.success, f.hdr.reason);
    else if (!terminalReached) setStatus(f.hdr.phase || "executing", f.hdr.detail || "");
    if (f.hdr.term || (lastDisplayedFid % 8) === 0){
      send({type:"client_perf", decode_ms: 0, draw_ms: 0, gap_ms: f.gap || 0});
    }
    decoding = false;
  }).catch(() => { decoding = false; setStatus("failed", "画面解码失败", "bad"); });
}

suiteSel.onchange = fillTasks;
taskSel.onchange = applyTask;
loadBtn.onclick = () => {
  if (!wsConnected) return;
  if (!suiteSel.value) return;
  loadBtn.disabled = true; startBtn.disabled = true; manualBtn.disabled = true; longBtn.disabled = true; restartBtn.disabled = true;
  taskIn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  setConfirm(false);
  phAgent.style.display = ""; phWrist.style.display = "";
  setStatus("加载中", `${suiteSel.value} t${taskSel.value} seed ${seedIn.value}`);
  send({type:"load", suite: suiteSel.value, task_id: Number(taskSel.value), seed: Number(seedIn.value || 0)});
};
function begin(source){
  if (!wsConnected) return;
  const task = taskIn.value.trim();
  if (!task) return;
  startBtn.disabled = true; manualBtn.disabled = true; longBtn.disabled = true; restartBtn.disabled = true; loadBtn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  setConfirm(false);
  stopBtn.disabled = false;
  const planSource = source === "model" ? "model" : "manual";
  const manualScope = source === "long" ? "long" : "single";
  send({type: source === "restart" ? "restart" : "start", task, plan_source: planSource, manual_scope: manualScope});
  setStatus(planSource === "manual" ? "人工计划" : "planning", planSource === "manual" ? `plan_source=manual ${manualScope}` : "等待模型");
}
startBtn.onclick = () => begin("model");
manualBtn.onclick = () => begin("manual");
longBtn.onclick = () => begin("long");
restartBtn.onclick = () => begin("restart");
stopBtn.onclick = () => { send({type:"stop"}); setConfirm(false); stopBtn.disabled = true; };
function confirm(verdict){
  if (!waiting) return;
  send({type:"confirm", verdict, episode_id: episode, scene_epoch: sceneEpoch, plan_version: planVersion});
  setConfirm(false);
  setStatus("executing", "已提交确认");
}
okBtn.onclick = () => confirm("success");
badBtn.onclick = () => confirm("failure");
unkBtn.onclick = () => confirm("unknown");
$("planner-expand").onclick = () => { plannerFull.hidden = !plannerFull.hidden; };
taskIn.addEventListener("input", ()=>{
  const ready = !!taskIn.value.trim();
  manualBtn.disabled = !ready;
  startBtn.disabled = !ready || !(lastHealth && lastHealth.can_model_plan);
});
async function bootstrap(){
  healthEl.textContent = "正在检查服务…";
  $("retry").hidden = true;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 4000);
  const started = performance.now();
  try {
    const response = await fetch("/health", {signal: ctrl.signal, cache: "no-store"});
    const text = await response.text();
    let body;
    try { body = JSON.parse(text); }
    catch (err) { throw new Error(`健康检查不是 JSON：HTTP ${response.status} ${text.slice(0, 180)}`); }
    if (!response.ok) throw new Error(`健康检查 HTTP ${response.status}`);
    renderHealth(body);
    setStatus("适配器在线", `${Math.round(performance.now() - started)}ms · 可以选择套件，环境在点击加载后才启动`);
    await refreshSuites();
  } catch (err) {
    healthEl.textContent = "检查超时。适配器没有在时限内返回，没有改跑干跑。";
    healthEl.style.color = "var(--bad)";
    $("retry").hidden = false;
    setStatus("failed", err.name === "AbortError" ? "健康检查超过 4 秒" : String(err.message || err), "bad");
  } finally {
    clearTimeout(timer);
  }
}
async function refreshSuites(){
  const response = await fetch("/suites", {cache: "no-store"});
  const body = await response.json();
  if (body.status === "ready") {
    applyCatalog(body.suites || []);
    return;
  }
  if (body.status === "failed") {
    setStatus("failed", body.detail || "读取套件失败", "bad");
    $("retry").hidden = false;
    return;
  }
  setStatus("套件加载中", "正在从 LIBERO 注册表读取，不会启动环境");
  if (!suitePoll) suitePoll = setInterval(refreshSuites, 1000);
}
$("retry").onclick = () => bootstrap();
setInterval(()=>{ send({type:"ping"}); }, 20000);
connect();
bootstrap();
requestAnimationFrame(tick);
