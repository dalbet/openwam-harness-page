"use strict";
// ---- Page: Show-Harness ManiSkill 闭环 UI 客户端 ----
const $ = (id) => document.getElementById(id);
const sceneSel = $("scene"), loadBtn = $("load"), taskIn = $("task"),
  startBtn = $("start"), restartBtn = $("restart"), stopBtn = $("stop"), resetBtn = $("reset"),
  agentCanvas = $("agent"), wristCanvas = $("wrist"),
  phAgent = $("ph-agent"), phWrist = $("ph-wrist"), statusEl = $("status"),
  idAgent = $("id-agent"), idAgentMode = $("id-agent-mode"), idAction = $("id-action"), idWorld = $("id-world"),
  idInference = $("id-inference"), idPhase = $("id-phase"),
  idFailureWrap = $("id-failure-wrap"), idFailure = $("id-failure"),
  agentModeSel = $("agent-mode"), switchAgentBtn = $("switch-agent");

const agCtx = agentCanvas.getContext("2d");
const wrCtx = wristCanvas.getContext("2d");

// 新画面稀疏（每个控制步一帧，决策之间有模型等待）。不凑 3 帧。
let pending = null;    // 最多保留一帧待绘制；终止帧不被普通帧覆盖
let decoding = false;
let curEid = null;
let acceptEid = null;
let gate = false;
let lastDisplayedFid = -1;
let terminalReached = false;
let lastArrival = 0;
let endedFallback = null;

function clearBuffer(){
  pending = null; decoding = false; lastDisplayedFid = -1; endedFallback = null;
}
function fitCanvas(canvas, w, h){
  if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)){
    canvas.width = w; canvas.height = h;
  }
}
function setStatus(phase, detail, cls){
  const c = cls ? ` ${cls}` : "";
  statusEl.textContent = "";
  const span = document.createElement("span");
  span.className = `phase${c}`;
  span.textContent = phase || "";
  statusEl.appendChild(span);
  if (detail){ statusEl.appendChild(document.createTextNode(` — ${detail}`)); }
}
const IDLE_PHASES = new Set(["starting", "ready", "succeeded", "failed", "stopped"]);
let agentMode = null;
function refreshAgentSwitch(){
  const idle = IDLE_PHASES.has(idPhase.textContent);
  switchAgentBtn.disabled = !wsConnected || !idle || !agentModeSel.value || agentModeSel.value === agentMode;
  switchAgentBtn.title = idle ? "" : "仅空闲状态可切换 Agent";
}
function applyIdentity(m){
  if (m.agent_model) idAgent.textContent = m.agent_model;
  if (m.agent_mode){
    agentMode = m.agent_mode;
    idAgentMode.textContent = m.agent_mode;
    if (m.agent_modes && !agentModeSel.options.length){
      m.agent_modes.forEach((mode)=>{
        const o = document.createElement("option");
        o.value = mode; o.textContent = mode;
        agentModeSel.appendChild(o);
      });
    }
    if (!agentModeSel.value) agentModeSel.value = m.agent_mode;
  }
  if (m.action_model) idAction.textContent = `${m.action_model}${m.action_backend ? ` (${m.action_backend})` : ""}`;
  if (m.world_model) idWorld.textContent = `${m.world_model} [${m.world_model_status || "unknown"}]`;
  if (m.inference_mode) idInference.textContent = `${m.inference_mode}${m.inference_backend ? ` / ${m.inference_backend}` : ""}`;
  if (m.phase) idPhase.textContent = m.phase;
  const failure = (m.failure_reason || "").trim();
  idFailureWrap.style.display = failure ? "" : "none";
  idFailure.textContent = failure;
  refreshAgentSwitch();
}

// ---- WebSocket ----
let ws = null, wsConnected = false;
function connect(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsConnected = true; setStatus("就绪", "已连接服务器"); requestScenes(); refreshAgentSwitch(); };
  ws.onclose = () => {
    wsConnected = false;
    setStatus("断开", "与服务器的连接已断开，尝试重连…", "warn");
    disableControls(true);
    clearBuffer();
    refreshAgentSwitch();
    setTimeout(connect, 1500);
  };
  ws.onerror = () => { try{ ws.close(); }catch(e){} };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string"){ handleText(ev.data); }
    else { handleBinary(ev.data); }
  };
}
function send(obj){ if(ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
function requestScenes(){ send({type:"get_scenes"}); }

// ---- 文本控制消息 ----
function handleText(txt){
  let m; try{ m = JSON.parse(txt); }catch(e){ return; }
  switch(m.type){
    case "scenes":
      sceneSel.innerHTML = "";
      (m.scenes||[]).forEach(s=>{
        const o = document.createElement("option");
        o.value = s.id; o.textContent = s.label;
        sceneSel.appendChild(o);
      });
      break;
    case "loaded":
      curEid = null; acceptEid = null; gate = false; clearBuffer();
      taskIn.value = m.instruction || "";
      taskIn.disabled = false; startBtn.disabled = false; restartBtn.disabled = true;
      stopBtn.disabled = true; resetBtn.disabled = false;
      loadBtn.disabled = false;
      setStatus("场景已加载", m.env_id, "good");
      phAgent.textContent = "等待开始"; phWrist.textContent = "等待开始";
      break;
    case "started":
      acceptEid = m.episode_id; gate = false; curEid = m.episode_id;
      clearBuffer(); terminalReached = false;
      stopBtn.disabled = false;
      break;
    case "identity":
      applyIdentity(m);
      break;
    case "load_failed":
      loadBtn.disabled = false;
      setStatus("加载失败", m.detail || "", "bad");
      break;
    case "state":
      // 临时状态（加载中/准备中/运行中），不覆盖终止状态
      if(!terminalReached) setStatus(m.phase, m.detail||"", m.cls||"");
      if (m.phase) idPhase.textContent = m.phase;
      if (m.cls === "bad" && m.detail){ idFailureWrap.style.display = ""; idFailure.textContent = m.detail; }
      refreshAgentSwitch();
      break;
    case "ended":
      if (acceptEid && m.episode_id && m.episode_id !== acceptEid) break;
      endedFallback = m;
      stopBtn.disabled = true; resetBtn.disabled = false;
      break;
    case "pong": break;
  }
}

// ---- 二进制帧 ----
function handleBinary(ab){
  const u8 = new Uint8Array(ab);
  const dv = new DataView(ab);
  const hlen = dv.getUint32(0);
  const hdrJson = new TextDecoder().decode(u8.subarray(4, 4+hlen));
  let hdr; try{ hdr = JSON.parse(hdrJson); }catch(e){ return; }
  if (gate) return;
  if (acceptEid !== null && hdr.eid !== acceptEid) return;
  let off = 4 + hlen;
  const aLen = hdr.a_len|0, wLen = hdr.w_len|0;
  const aBlob = u8.subarray(off, off+aLen); off += aLen;
  const wBlob = u8.subarray(off, off+wLen);
  // episode 变更 -> 清空旧缓冲
  if (curEid !== null && hdr.eid !== curEid){ clearBuffer(); }
  curEid = hdr.eid;
  if (!hdr.term) terminalReached = false;
  const now = performance.now();
  const gap = lastArrival ? now - lastArrival : 0;
  lastArrival = now;
  const frame = { hdr, aBlob, wBlob, arrival: now, gap };
  if (hdr.term) pending = frame;
  else if (!pending || !pending.hdr.term) pending = frame;
}

function showResult(success, reason){
  terminalReached = true;
  setStatus(success ? "完成" : "失败", reason || "", success ? "good" : "bad");
  startBtn.disabled = false; restartBtn.disabled = false; loadBtn.disabled = false;
  stopBtn.disabled = true; resetBtn.disabled = false;
  if (!success && reason){ idFailureWrap.style.display = ""; idFailure.textContent = reason; }
  endedFallback = null;
}

function tick(){
  requestAnimationFrame(tick);
  if (endedFallback && !pending && !decoding && performance.now() - (endedFallback._at || (endedFallback._at = performance.now())) > 1500){
    showResult(!!endedFallback.success, endedFallback.reason);
    return;
  }
  if (decoding || !pending) return;
  const f = pending;
  pending = null;
  decoding = true;
  const t0 = performance.now();
  const aBlob = new Blob([f.aBlob], {type:"image/jpeg"});
  const wBlob = new Blob([f.wBlob], {type:"image/jpeg"});
  Promise.all([createImageBitmap(aBlob), createImageBitmap(wBlob)]).then(([aBmp, wBmp]) => {
    const t1 = performance.now();
    fitCanvas(agentCanvas, aBmp.width, aBmp.height);
    fitCanvas(wristCanvas, wBmp.width, wBmp.height);
    agCtx.drawImage(aBmp, 0, 0);
    wrCtx.drawImage(wBmp, 0, 0);
    aBmp.close(); wBmp.close();
    const t2 = performance.now();
    phAgent.style.display = "none"; phWrist.style.display = "none";
    lastDisplayedFid = f.hdr.fid;
    const h = f.hdr;
    if (h.term) showResult(!!h.success, h.reason);
    else if (!terminalReached) setStatus(h.phase || "运行中", h.detail || "", "");
    if (h.term || (lastDisplayedFid % 8) === 0){
      send({type:"client_perf", decode_ms: t1 - t0, draw_ms: t2 - t1, gap_ms: f.gap || 0});
    }
    decoding = false;
  }).catch(() => { decoding = false; });
}

// ---- 控件 ----
function disableControls(disabled){
  startBtn.disabled = disabled || !taskIn.value;
  // load/restart 由状态机单独控制
}
loadBtn.onclick = () => {
  if (!wsConnected) return;
  const id = sceneSel.value;
  if (!id) return;
  loadBtn.disabled = true; startBtn.disabled = true; restartBtn.disabled = true;
  stopBtn.disabled = true; resetBtn.disabled = true;
  taskIn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  setStatus("加载中", id);
  send({type:"load", scene:id});
};
startBtn.onclick = () => {
  if (!wsConnected) return;
  const task = taskIn.value.trim();
  if (!task) return;
  startBtn.disabled = true; restartBtn.disabled = true; loadBtn.disabled = true;
  stopBtn.disabled = false;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = true;
  send({type:"start", task});
  setStatus("准备中", "首次推理可能需要等待模型加载…", "warn");
};
restartBtn.onclick = () => {
  if (!wsConnected) return;
  const task = taskIn.value.trim();
  if (!task) return;
  startBtn.disabled = true; restartBtn.disabled = true; loadBtn.disabled = true;
  stopBtn.disabled = false;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = true;
  send({type:"restart", task});
  setStatus("准备中", "重置环境并重新执行…", "warn");
};
stopBtn.onclick = () => {
  if (!wsConnected) return;
  stopBtn.disabled = true;
  send({type:"stop"});
  setStatus("停止中", "等待当前动作结束…", "warn");
};
resetBtn.onclick = () => {
  if (!wsConnected) return;
  resetBtn.disabled = true; startBtn.disabled = true; restartBtn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  send({type:"reset"});
  setStatus("重置中", "重新初始化当前场景…", "warn");
};
switchAgentBtn.onclick = () => {
  if (!wsConnected || !agentModeSel.value) return;
  switchAgentBtn.disabled = true;
  send({type:"agent_mode", mode:agentModeSel.value});
  setStatus("切换 Agent", agentModeSel.value, "warn");
};
agentModeSel.addEventListener("change", refreshAgentSwitch);
taskIn.addEventListener("input", ()=>{ startBtn.disabled = !taskIn.value.trim(); });

// 心跳
setInterval(()=>{ send({type:"ping"}); }, 20000);
connect();
requestAnimationFrame(tick);
