"use strict";
// ---- Page: 三层具身引擎页面（场景 / 任务 / 方案 + 双路实时画面） ----
const $ = (id) => document.getElementById(id);
const sceneSel = $("scene"), loadBtn = $("load"), taskIn = $("task"),
  startBtn = $("start"), restartBtn = $("restart"), stopBtn = $("stop"), resetBtn = $("reset"),
  agentCanvas = $("agent"), wristCanvas = $("wrist"),
  phAgent = $("ph-agent"), phWrist = $("ph-wrist"), statusEl = $("status"),
  idArm = $("id-arm"), idServing = $("id-serving"),
  idAgent = $("id-agent"), idAgentMode = $("id-agent-mode"), idAction = $("id-action"), idWorld = $("id-world"),
  idInference = $("id-inference"), idPhase = $("id-phase"),
  idFailureWrap = $("id-failure-wrap"), idFailure = $("id-failure"),
  idRaw = $("id-raw"), idChoice = $("id-choice"), idStandard = $("id-standard"),
  idSwitchWrap = $("id-switch-wrap"), idSwitch = $("id-switch"),
  armCtrl = $("arm-ctrl"), armSel = $("arm"), armBtn = $("armBtn"),
  mTask = $("m-task"), mProgress = $("m-progress"), mPlan = $("m-plan"),
  mFirst = $("m-first"), mGen = $("m-gen"), mInfer = $("m-infer"),
  mDur = $("m-dur"), mFrame = $("m-frame"), logEl = $("log");

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
let framesShown = 0;
let framesOverwritten = 0;

let sceneRows = [];
let armState = null;
let metricsState = null;
let armSwitching = false;
let localBusy = false;   // set on click, cleared when the engine reports a terminal/idle phase

const BUSY_PHASES = new Set(["loading", "planning", "running", "recovering", "verifying", "stopping"]);
const ACTIVE_PHASES = new Set(["planning", "running", "recovering", "verifying", "stopping"]);

function logLine(kind, text){
  const row = `${new Date().toLocaleTimeString()} · ${kind} · ${text}`;
  const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 4;
  logEl.textContent += (logEl.textContent ? "\n" : "") + row;
  const lines = logEl.textContent.split("\n");
  if (lines.length > 200) logEl.textContent = lines.slice(-200).join("\n");
  if (atBottom) logEl.scrollTop = logEl.scrollHeight;
}

function clearBuffer(){
  pending = null; decoding = false; lastDisplayedFid = -1; endedFallback = null;
}
function showWaiting(text){
  phAgent.style.display = ""; phWrist.style.display = "";
  phAgent.textContent = text; phWrist.textContent = text;
  agCtx.clearRect(0, 0, agentCanvas.width, agentCanvas.height);
  wrCtx.clearRect(0, 0, wristCanvas.width, wristCanvas.height);
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
function applyIdentity(m){
  if (m.agent_model) idAgent.textContent = `${m.agent_model}${m.agent_mode ? ` · ${m.agent_mode}` : ""}`;
  const router = m.agent_router || {};
  if (idAgentMode){
    idAgentMode.textContent = router.model_mode
      ? `${router.model_mode} / ${router.max_tokens} tok / catalog ${(router.catalog || {}).version || "?"}`
      : "semantic_router";
  }
  if (m.action_model) idAction.textContent = `${m.action_model}${m.action_backend ? ` (${m.action_backend})` : ""}`;
  if (m.world_model) idWorld.textContent = `${m.world_model} [${m.world_model_status || "unknown"}]`;
  if (m.inference_mode) idInference.textContent = `${m.inference_mode}${m.inference_backend ? ` / ${m.inference_backend}` : ""}`;
  if (m.phase) idPhase.textContent = m.phase;
  const failure = (m.failure_reason || "").trim();
  idFailureWrap.style.display = failure ? "" : "none";
  idFailure.textContent = failure;
  const u = m.understanding || {};
  idRaw.textContent = u.raw_instruction || m.raw_instruction || "—";
  if (u.semantic_choice){
    const task = u.task_id !== null && u.task_id !== undefined ? ` → task ${u.task_id} ${u.task_key || ""}` : "";
    idChoice.textContent = `${u.semantic_choice}${task}`;
    idChoice.className = u.decision_status === "ok" ? "good" : "bad";
  } else {
    idChoice.textContent = "—";
    idChoice.className = "";
  }
  idStandard.textContent = u.official_instruction || "—";
  idStandard.className = u.official_instruction ? "" : "bad";
  const needSwitch = u.needs_scene_switch === true;
  idSwitchWrap.style.display = needSwitch ? "" : "none";
  idSwitch.textContent = needSwitch
    ? `task ${u.required_task_id} ${u.required_task_key || ""}`
    : "";
}

// ---- 方案（arm）----
function armLabel(id){
  const row = (armState && (armState.arms || []).find(a => a.id === id));
  return row ? row.label : id;
}
function renderArm(){
  const m = armState;
  if (!m) return;
  if (!m.enabled){
    armCtrl.style.display = "none";
    return;
  }
  armCtrl.style.display = "";
  const rows = m.arms || [];
  if (armSel.options.length !== rows.length){
    armSel.innerHTML = "";
    rows.forEach(a => {
      const o = document.createElement("option");
      o.value = a.id; o.textContent = a.label;
      armSel.appendChild(o);
    });
  }
  if (m.selected) armSel.value = m.selected;
  const armed = !!m.selected && m.verified && m.actual === m.selected;
  if (!m.selected){
    idArm.textContent = `未选择（当前 ${m.actual_label}）`;
    idArm.className = "warn";
  } else if (armed){
    idArm.textContent = `${armLabel(m.selected)} · 已校验`;
    idArm.className = "good";
  } else if (m.switching){
    idArm.textContent = `${armLabel(m.switching)} · 切换中`;
    idArm.className = "warn";
  } else {
    idArm.textContent = `${armLabel(m.selected)} · 未生效（实际 ${m.actual_label}）`;
    idArm.className = "bad";
  }
  idServing.textContent = `${m.actual_label}${m.actual_backend ? ` · ${m.actual_backend}` : ""}`
    + `${m.actual_precision ? ` · ${m.actual_precision}` : ""}`
    + `${m.actual_server_url ? ` @ ${m.actual_server_url.replace(/^https?:\/\//, "")}` : ""}`;
  idServing.className = armed ? "good" : "warn";
  armSwitching = !!m.switching;
  armBtn.disabled = armSwitching || localBusy || !wsConnected || BUSY_PHASES.has(m.phase || "");
  if (m.error) logLine("方案", m.error);
}
function applyArmProgress(m){
  if (m.stage) {
    setStatus("方案切换中", `${m.stage}${m.detail ? ` — ${m.detail}` : ""}`, "warn");
    logLine("方案", `${m.stage} ${m.detail || ""}`);
  }
}

// ---- 实时指标 ----
function fmtS(v, digits){
  return (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Number(v).toFixed(digits ?? 2)}s`;
}
function fmtMs(v){
  return (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Math.round(Number(v))}ms`;
}
function taskLabel(id){
  const row = sceneRows.find(s => Number(s.task_id) === Number(id));
  return id === null || id === undefined ? "—" : `task ${id}${row ? ` · ${row.label}` : ""}`;
}
function renderMetrics(){
  const m = metricsState;
  if (!m) return;
  mTask.textContent = taskLabel(m.task_id);
  const done = m.executed_actions ?? m.sim_actions;
  mProgress.textContent = (m.sim_actions === null || m.sim_actions === undefined)
    ? "—" : `${m.sim_actions} 动作${done && done !== m.sim_actions ? ` / 本轮 ${done}` : ""}`;
  mPlan.textContent = m.planner_s === null || m.planner_s === undefined
    ? "—" : `${fmtS(m.planner_s)}${m.planner_source ? ` · ${m.planner_source}` : ""}`;
  mFirst.textContent = fmtMs(m.first_action_s === null || m.first_action_s === undefined ? null : m.first_action_s * 1000);
  const g = m.generation || {};
  mGen.textContent = (g.n && g.server_median_ms !== null && g.server_median_ms !== undefined)
    ? `median ${Math.round(g.server_median_ms)}ms · p95 ${Math.round(g.server_p95_ms)}ms · n=${g.n}`
    : (g.n ? `n=${g.n}（无可比延迟）` : "—");
  mInfer.textContent = fmtS(m.inference_wait_s);
  // 任务耗时：运行中按引擎给出的起点实时计时；结束后用本轮记录的 duration_s
  let dur = m.duration_s;
  if (BUSY_PHASES.has(m.phase || "") && m.episode_started_ts){
    dur = (Date.now() / 1000) - Number(m.episode_started_ts);
  }
  mDur.textContent = fmtS(dur);
}
function renderFrameMeta(){
  const m = metricsState;
  if (!m) { mFrame.textContent = "—"; return; }
  const seq = m.frame_seq;
  const age = m.frame_ts ? Math.max(0, Date.now() / 1000 - Number(m.frame_ts)) : null;
  mFrame.textContent = (seq === null || seq === undefined)
    ? "—" : `#${seq} · ${fmtMs(age === null ? null : age * 1000)} 前${framesOverwritten ? ` · 覆盖 ${framesOverwritten}` : ""}`;
}

// ---- WebSocket ----
let ws = null, wsConnected = false;
function connect(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsConnected = true; setStatus("就绪", "已连接服务器"); requestScenes(); send({type:"get_arms"}); };
  ws.onclose = () => {
    wsConnected = false;
    setStatus("断开", "与服务器的连接已断开，尝试重连…", "warn");
    disableControls(true);
    clearBuffer();
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
      sceneRows = m.scenes || [];
      sceneSel.innerHTML = "";
      sceneRows.forEach(s=>{
        const o = document.createElement("option");
        o.value = s.id; o.textContent = s.label;
        sceneSel.appendChild(o);
      });
      renderMetrics();
      break;
    case "loaded":
      curEid = null; acceptEid = null; gate = false; clearBuffer();
      taskIn.value = m.instruction || "";
      taskIn.disabled = false; startBtn.disabled = false; restartBtn.disabled = true;
      stopBtn.disabled = true; resetBtn.disabled = false;
      loadBtn.disabled = false;
      localBusy = false;
      idFailureWrap.style.display = "none";
      setStatus("场景已加载", m.env_id, "good");
      showWaiting("等待开始");
      logLine("场景", `已加载 ${m.env_id || ""}`);
      break;
    case "started":
      acceptEid = m.episode_id; gate = false; curEid = m.episode_id;
      clearBuffer(); terminalReached = false;
      stopBtn.disabled = false;
      showWaiting("等待首帧…");
      logLine("回合", `开始 ${m.episode_id || ""}`);
      break;
    case "identity":
      applyIdentity(m);
      break;
    case "arm_state":
      armState = m;
      renderArm();
      if (m.status === "error" && m.error){ logLine("方案", `失败：${m.error}`); }
      break;
    case "arm_progress":
      applyArmProgress(m);
      break;
    case "metrics":
      metricsState = m;
      renderMetrics();
      renderFrameMeta();
      break;
    case "load_failed":
      loadBtn.disabled = false;
      setStatus("加载失败", m.detail || "", "bad");
      logLine("场景", `加载失败 ${m.detail || ""}`);
      break;
    case "state":
      // 临时状态（加载中/准备中/运行中），不覆盖终止状态
      if(!terminalReached) setStatus(m.phase, m.detail||"", m.cls||"");
      if (m.phase) idPhase.textContent = m.phase;
      if (m.cls === "bad" && m.detail && m.kind !== "arm"){
        idFailureWrap.style.display = ""; idFailure.textContent = m.detail;
        // 启动被拒绝时不能把页面留在“准备中”且吞掉画面
        if (gate){ gate = false; startBtn.disabled = !taskIn.value.trim(); loadBtn.disabled = false; }
        logLine("状态", `${m.phase} — ${m.detail}`);
      }
      break;
    case "ended":
      localBusy = false;
      if (acceptEid && m.episode_id && m.episode_id !== acceptEid) break;
      endedFallback = m;
      stopBtn.disabled = true; resetBtn.disabled = false;
      logLine("回合", `${m.success ? "完成" : "结束"} ${m.reason || ""}`);
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
  else if (!pending || !pending.hdr.term){ if (pending) framesOverwritten += 1; pending = frame; }
}

function showResult(success, reason){
  terminalReached = true;
  localBusy = false;
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
    framesShown += 1;
    const h = f.hdr;
    if (h.term) showResult(!!h.success, h.reason);
    else if (!terminalReached) setStatus(h.phase || "运行中", h.detail || "", "");
    if (h.term || (lastDisplayedFid % 8) === 0){
      send({type:"client_perf", decode_ms: t1 - t0, draw_ms: t2 - t1, gap_ms: f.gap || 0});
    }
    decoding = false;
    renderFrameMeta();
  }).catch(() => { decoding = false; });
}

// ---- 控件 ----
function disableControls(disabled){
  startBtn.disabled = disabled || !taskIn.value;
  // load/restart 由状态机单独控制
}
function busy(){ return armSwitching || localBusy || BUSY_PHASES.has((armState && armState.phase) || ""); }

loadBtn.onclick = () => {
  if (!wsConnected || busy()) return;
  const id = sceneSel.value;
  if (!id) return;
  loadBtn.disabled = true; startBtn.disabled = true; restartBtn.disabled = true;
  stopBtn.disabled = true; resetBtn.disabled = true;
  taskIn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  showWaiting("加载中…");
  localBusy = true;
  setStatus("加载中", id);
  send({type:"load", scene:id});
};
startBtn.onclick = () => {
  if (!wsConnected || busy()) return;
  const task = taskIn.value.trim();
  if (!task) return;
  startBtn.disabled = true; restartBtn.disabled = true; loadBtn.disabled = true;
  stopBtn.disabled = false;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = true;
  framesShown = 0; framesOverwritten = 0;
  showWaiting("等待首帧…");
  localBusy = true;
  send({type:"start", task});
  setStatus("准备中", "首次推理可能需要等待模型加载…", "warn");
};
restartBtn.onclick = () => {
  if (!wsConnected || busy()) return;
  const task = taskIn.value.trim();
  if (!task) return;
  startBtn.disabled = true; restartBtn.disabled = true; loadBtn.disabled = true;
  stopBtn.disabled = false;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = true;
  framesShown = 0; framesOverwritten = 0;
  showWaiting("等待首帧…");
  localBusy = true;
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
  if (!wsConnected || busy()) return;
  resetBtn.disabled = true; startBtn.disabled = true; restartBtn.disabled = true;
  clearBuffer(); terminalReached = false; curEid = null; acceptEid = null; gate = false;
  showWaiting("重置中…");
  localBusy = true;
  send({type:"reset"});
  setStatus("重置中", "重新初始化当前场景…", "warn");
};
armBtn.onclick = () => {
  if (!wsConnected || armSwitching) return;
  const arm = armSel.value;
  if (!arm) return;
  armBtn.disabled = true;
  setStatus("方案切换中", `请求切换到 ${armLabel(arm)}`, "warn");
  logLine("方案", `请求切换 ${arm}`);
  send({type:"set_arm", arm});
};
taskIn.addEventListener("input", ()=>{ startBtn.disabled = !taskIn.value.trim() || busy(); });

// 心跳 + 实时指标刷新（任务耗时每秒走表）
setInterval(()=>{ send({type:"ping"}); }, 20000);
setInterval(renderMetrics, 500);
connect();
requestAnimationFrame(tick);
