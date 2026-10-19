"use strict";
const $ = (id) => document.getElementById(id);
const suiteSel = $("suite"), modeSel = $("execution-mode"), progressEl = $("progress"),
  loadBtn = $("load"), taskIn = $("task"), profileSel = $("planner-profile"),
  startBtn = $("start"),
  restartBtn = $("restart"), stopBtn = $("stop"),
  okBtn = $("ok"), badBtn = $("bad"), unkBtn = $("unk"),
  confirmBox = $("confirm"), goalEl = $("goal"),
  agentCanvas = $("agent"), wristCanvas = $("wrist"),
  agentView = $("agent-view"), wristView = $("wrist-view"),
  phAgent = $("ph-agent"), phWrist = $("ph-wrist"), statusEl = $("status"),
  healthEl = $("health"),
  plannerMeta = $("planner-meta"), plannerBrief = $("planner-brief"), plannerFull = $("planner-full"),
  execMeta = $("exec-meta"), execBrief = $("exec-brief"), execFull = $("exec-full");

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
let lastPlannerStage = "";
let lastExec = null;
let lastRunStatus = "";
let framesReceived = 0;
let framesDrawn = 0;
let maxFrameGapMs = 0;
let maxFrameAgeMs = 0;
let lastFrameMeta = null;
let rootError = "";
let lastSnapshot = null;
let serverInstance = "";
let stateRevision = -1;
let clockOffset = null;
let taskEdited = false;
const drawnFrameIds = new Set();

function plannerReady(){
  return !!(lastHealth && lastHealth.can_model_plan);
}
function syncRunButtons(){
  const connected = !!(ws && ws.readyState === 1);
  let reason = "正在同步";
  if (!connected) reason = "未连接";
  else if (lastSnapshot) reason = lastSnapshot.block_reason || lastSnapshot.start_block || "";
  startBtn.disabled = !!reason;
  startBtn.textContent = "开始";
  const reasonEl = $("start-reason");
  if (reasonEl) reasonEl.textContent = reason;
  startBtn.title = reason;
  if (loadBtn && lastSnapshot) loadBtn.disabled = !connected || !lastSnapshot.can_load;
  if (stopBtn && lastSnapshot) stopBtn.disabled = !connected || !lastSnapshot.can_stop;
}
function noteInstance(m){
  if (!m || !m.server_instance_id) return true;
  if (serverInstance && m.server_instance_id !== serverInstance) {
    serverInstance = m.server_instance_id;
    stateRevision = -1;
    sceneEpoch = null;
    episode = null;
    lastSnapshot = null;
    rootError = "";
    send({type:"get_snapshot"});
    return false;
  }
  serverInstance = m.server_instance_id;
  return true;
}
function acceptControl(m){
  if (!noteInstance(m)) return false;
  if (typeof m.state_revision === "number") {
    if (m.state_revision < stateRevision) return false;
    stateRevision = m.state_revision;
  }
  if (typeof m.scene_epoch === "number" && sceneEpoch !== null && m.scene_epoch < sceneEpoch) return false;
  if (m.type === "ended") {
    if (m.scene_epoch != null && sceneEpoch !== null && Number(m.scene_epoch) !== Number(sceneEpoch)) return false;
    if (episode && m.episode_id && m.episode_id !== episode) return false;
    const currentRun = lastSnapshot && lastSnapshot.run_id;
    if (currentRun && m.run_id && m.run_id < currentRun) return false;
  }
  return true;
}

function clearBuffer(){
  pending = null; decoding = false; lastDisplayedFid = -1; endedFallback = null;
}
function setStatus(phase, detail, cls){
  const text = rootError || detail || "";
  const c = cls ? ` ${cls}` : "";
  statusEl.innerHTML = `<span class="phase${c}">${escapeHtml(phase)}</span>` + (text ? ` — ${escapeHtml(text)}` : "");
}
function escapeHtml(value){
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[ch]));
}
function setConfirm(on){
  waiting = on;
  confirmBox.hidden = !on;
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
  planning: "规划中",
  checking_task: "检查任务",
  binding: "绑定对象",
  validating: "校验中",
  switching: "切换任务",
  loading: "加载中",
  loaded: "场景已加载",
  plan_ready: "计划已生成",
  executing: "执行中",
  waiting_confirmation: "等待确认",
  failed: "失败"
};
let suites = [];
let ws = null, wsConnected = false;
let lastHealth = null;
let suitePoll = null;
function connect(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsConnected = true; if (!rootError) setStatus("就绪", location.host); send({type:"get_scenes"}); send({type:"get_health"}); send({type:"get_snapshot"}); };
  ws.onclose = () => {
    wsConnected = false;
    setStatus("断开", rootError || "与服务器的连接已断开，尝试重连…", rootError ? "bad" : "warn");
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
  if (!acceptControl(m)) return;
  switch(m.type){
    case "health":
      renderHealth(m);
      break;
    case "snapshot":
      applySnapshot(m);
      break;
    case "load_ack":
    case "load_rejected":
      if (m.status === "accepted") break;
      if (!rootError && !(lastSnapshot && lastSnapshot.block_reason)) setStatus("已拒绝", m.detail || "不能加载", "warn");
      break;
    case "suites":
      if (m.status === "ready") applyCatalog(m.suites || []);
      break;
    case "loaded":
      rootError = "";
      curEid = null; acceptEid = null; gate = false; clearBuffer();
      episode = m.episode_id; sceneEpoch = m.scene_epoch; planVersion = m.plan_version || 1;
      goalEl.textContent = m.summary_zh ? `当前目标：${m.summary_zh}` : `当前目标：${m.instruction || ""}`;
      taskIn.disabled = false;
      restartBtn.disabled = false;
      setConfirm(false);
      if (m.placeholder || m.mode === "dry"){
        setStatus("干跑", "未创建真实环境，没有新图像", "bad");
        phAgent.textContent = "干跑，无图像"; phWrist.textContent = "干跑，无图像";
      } else {
        const scene = m.scene || {};
        const loadedDetail = `${m.suite || scene.suite || ""} task ${m.task_id ?? scene.task_id ?? "?"} seed ${m.seed ?? scene.seed ?? "?"} · step ${scene.env_step ?? "?"}`;
        const plannerNote = plannerReady() ? "" : " · Planner 未连接";
        setStatus("场景已加载", loadedDetail + plannerNote, plannerReady() ? "good" : "warn");
        phAgent.textContent = "等待画面"; phWrist.textContent = "等待画面";
      }
      break;
    case "start_ack":
      if (m.status === "rejected") {
        if (!rootError) setStatus("已拒绝", m.detail || m.code || "开始被拒绝", "bad");
        break;
      }
      rootError = "";
      if (m.used_loaded_goal) goalEl.textContent = `实际执行：${m.summary_zh || m.goal || ""}`;
      else goalEl.textContent = `输入保留：${m.user_text || ""} · 规范：${m.summary_zh || m.goal || ""}`;
      setStatus("检查任务", m.client_request_id || "");
      break;
    case "task_bound":
      if (m.used_loaded_goal) goalEl.textContent = `实际执行：${m.summary_zh || m.goal || ""}`;
      else goalEl.textContent = `输入保留：${m.user_text || ""} · 规范：${m.summary_zh || m.goal || ""}`;
      break;
    case "run_stage":
      if (m.stage === "planner_request" || m.stage === "planner_response") setStatus("规划中", m.detail || m.code || "", m.status === "failed" || m.status === "rejected" ? "bad" : "");
      else if (m.stage === "validate" || m.stage === "compile") setStatus("校验中", m.detail || "", m.status === "failed" ? "bad" : "");
      else if (m.stage === "first_tool") setStatus("执行中", m.detail || "");
      else if (m.stage === "binding") setStatus("绑定对象", m.detail || "");
      else if (m.stage === "task_check") setStatus("检查任务", m.detail || "", m.status === "rejected" ? "bad" : "");
      break;
    case "started":
      acceptEid = m.episode_id; gate = false; curEid = m.episode_id;
      episode = m.episode_id;
      clearBuffer(); terminalReached = false;
      lastExec = null; framesReceived = 0; framesDrawn = 0; maxFrameGapMs = 0; maxFrameAgeMs = 0; lastFrameMeta = null;
      drawnFrameIds.clear();
      clockOffset = null;
      updateMotion();
      stopBtn.disabled = false;
      setStatus(m.mode === "dry" ? "干跑" : "planning", m.plan_source || "");
      break;
    case "load_failed":
      taskIn.disabled = !episode;
      setStatus("failed", m.detail || "加载失败", "bad");
      break;
    case "state":
      if (m.scene_epoch != null) sceneEpoch = m.scene_epoch;
      if (m.plan_version != null) planVersion = m.plan_version;
      if (m.phase === "waiting_confirmation") setConfirm(true);
      else if (m.phase === "failed" || m.phase === "executing" || m.phase === "planning") setConfirm(false);
      if (m.execution) rememberExec(m.execution);
      if (m.planner_stage) lastPlannerStage = m.planner_stage;
      if ((m.phase === "failed" || m.phase === "waiting_confirmation") && !rootError) {
        rootError = (m.root_error || (m.execution && m.execution.reason) || m.reason || m.detail || "");
      }
      if (!terminalReached && lastPlannerStage === "ok" && m.phase === "failed") {
        setStatus("规划通过，执行失败", (m.execution && m.execution.reason) || m.reason || m.detail || "", "bad");
      } else if (!terminalReached && lastPlannerStage === "ok" && m.phase === "waiting_confirmation") {
        setStatus("规划通过，执行暂停", (m.execution && m.execution.reason) || m.detail || "", "warn");
      } else if (!terminalReached) setStatus(PHASES[m.phase] || m.phase || "state", statusDetail(m), m.phase === "failed" ? "bad" : "");
      if (m.phase === "executing" && progressEl) progressEl.textContent = `子目标：${m.subtask || m.step_id || m.detail || "执行中"}`;
      break;
    case "planner_io":
      renderPlanner(m);
      break;
    case "binding":
      renderBinding(m.binding || {});
      break;
    case "log":
      $("log").textContent += (m.text || "") + "\n";
      break;
    case "clarify":
      setStatus("需要确认", m.question || "请说明要哪一个物体", "warn");
      break;
    case "ended":
      if (acceptEid && m.episode_id && m.episode_id !== acceptEid) break;
      if (m.planner_stage) lastPlannerStage = m.planner_stage;
      if (m.status) lastRunStatus = m.status;
      if (m.execution) rememberExec(m.execution);
      if (progressEl && (m.native_task_success != null || m.user_goal_success != null)) {
        const nativeText = m.native_task_success ? "原生目标已达成" : "原生目标未达成";
        const userText = m.user_goal_success == null ? "" : (m.user_goal_success ? "用户目标已完成" : "用户目标未完成");
        progressEl.textContent = [m.execution_mode === "free" ? "自由模式" : "", nativeText, userText].filter(Boolean).join(" · ");
      }
      if (!rootError && m.success !== true) rootError = m.root_error || m.reason || "";
      endedFallback = m;
      setConfirm(false);
      restartBtn.disabled = false;
      break;
    case "pong": break;
  }
}

const CATEGORIES = {
  service: "连接或超时",
  parameter: "参数或对象名",
  subtask_incomplete: "子任务未完成",
  tool_exception: "工具异常"
};
function rememberExec(execution){
  lastExec = execution || null;
  renderExec(lastExec);
  updateMotion();
}
function updateMotion(){
  const el = $("motion");
  if (!el) return;
  const parts = ["执行：同步"];
  if (lastHealth){
    const commit = (lastHealth.commit || "").slice(0, 7) || "未知";
    parts.push(`commit ${commit}`, `前端 ${lastHealth.static_version || "未知"}`, `pid ${lastHealth.pid ?? "未知"}`);
  }
  const prim = (lastFrameMeta && lastFrameMeta.primitive) || (lastExec && lastExec.skill);
  if (prim) parts.push("当前 " + prim);
  if (lastFrameMeta && lastFrameMeta.frame_time) parts.push("最新帧 " + lastFrameMeta.frame_time);
  if (lastExec && lastExec.transport_started === false) parts.push("尚未搬运/放置");
  if (framesReceived || framesDrawn) parts.push(`收 ${framesReceived} / 绘 ${framesDrawn} / 不同帧 ${drawnFrameIds.size}`);
  el.textContent = parts.join(" · ");
}
function renderExec(execution){
  if (!execution) {
    execMeta.textContent = "执行：同步";
    execBrief.textContent = "";
    return;
  }
  const category = CATEGORIES[execution.category] || execution.category || "—";
  const checker = execution.checker_reason_code ? ` · 检查结果 ${execution.checker_reason_code}` : "";
  const planner = lastPlannerStage === "ok" ? "规划通过" : (lastPlannerStage || "规划状态未知");
  const pauseCodes = ["INSUFFICIENT_EVIDENCE", "NEED_STABILITY_SAMPLE", "TARGET_NOT_IDENTIFIED", "TARGET_AMBIGUOUS"];
  const paused = pauseCodes.includes(execution.reason_code);
  const failed = !paused && execution.reason_code && execution.reason_code !== "GRASP_HELD" && execution.reason_code !== "OBJECT_LOCALIZED" && !execution.stopped;
  const execState = paused ? `${planner}，执行暂停` : (failed ? `${planner}，执行失败` : `${planner} · ${execution.reason_code || "执行中"}`);
  execMeta.textContent = `执行：同步 · ${execState}`;
  execBrief.textContent = [
    `当前步骤: ${execution.step_id || "—"}`,
    `当前工具: ${execution.skill || "—"}`,
    `失败类别: ${category}`,
    `原因: ${execution.reason || "—"}${checker}`,
    (execution.unmet || []).length ? `未满足: ${(execution.unmet || []).join("；")}` : "",
    (execution.missing || []).length ? `缺少: ${(execution.missing || []).join(", ")}` : "",
    execution.transport_started === false ? "尚未搬运/放置" : "",
    `环境动作数: ${execution.env_action_count ?? 0}`,
    `最新观测: ${execution.observation_id || "—"}`,
    `记录序号: ${execution.record_index ?? "—"}（${execution.record_index_note || "不是仿真步数"}）`
  ].filter(Boolean).join("\n");
  execFull.hidden = true;
  execFull.textContent = JSON.stringify({
    args: execution.args || {},
    object_ref: execution.object_ref || null,
    grounding_query: execution.grounding_query || null,
    vla_prompt: execution.vla_prompt || null,
    return_summary: execution.return_summary || {},
    request_id: execution.request_id || null,
    episode_id: execution.episode_id || null,
    reason_code: execution.reason_code || null,
    missing: execution.missing || null,
    unmet: execution.unmet || null,
    transport_started: execution.transport_started
  }, null, 2);
}
function statusDetail(m){
  const timing = m.timings ? JSON.stringify(m.timings) : "";
  const bits = [m.detail || "", m.reason || "", m.reason_code || "", m.step_id || "", m.skill || "", m.evidence_source || "", timing].filter(Boolean);
  return bits.join(" · ");
}
function applyCatalog(items){
  suites = items || [];
  if (!suites.length) return;
  const previous = suiteSel.value;
  suiteSel.innerHTML = "";
  suites.forEach(suite => {
    const option = document.createElement("option");
    option.value = suite.id;
    option.textContent = suite.label || suite.id;
    suiteSel.appendChild(option);
  });
  if (previous && suites.some(item => item.id === previous)) suiteSel.value = previous;
  else if (suites.some(item => item.id === "libero_goal")) suiteSel.value = "libero_goal";
  suiteSel.disabled = false;
  loadBtn.disabled = false;
  if (suitePoll) { clearInterval(suitePoll); suitePoll = null; }
}

function renderHealth(m){
  lastHealth = m;
  const mode = m.mode === "live" ? "真实模式" : "干跑模式";
  const planner = m.planner === "ok" ? "planner 正常" : `planner ${m.planner || "未知"}`;
  const env = {ready: "env 就绪", loading: "env 加载中", down: "env 已停止", not_loaded: "env 未加载"}[m.env] || `env ${m.env || "未知"}`;
  const vla = {ready: "VLA 就绪", down: "VLA 已停止", not_loaded: "VLA 未加载"}[m.vla] || `VLA ${m.vla || "未知"}`;
  const commit = (m.commit || "").slice(0, 7) || "未知";
  healthEl.textContent = `适配器在线 · ${mode} · ${location.host} · pid ${m.pid} · commit ${commit} · 前端 ${m.static_version || "未知"} · ${planner} · ${env} · ${vla}`;
  healthEl.style.color = m.mode === "live" ? "var(--dim)" : "var(--warn)";
  if (m.error || m.planner_detail) healthEl.title = [m.error, m.planner_detail].filter(Boolean).join("\n");
  $("retry").hidden = true;
  syncRunButtons();
  updateMotion();
}

const STAGES = {
  request_failed: "请求失败",
  parse_failed: "解析失败",
  validation_failed: "校验失败",
  binding_unresolved: "对象待确认",
  ok: "通过"
};
function renderPlanner(m){
  const stage = STAGES[m.stage] || (m.plan_source ? "通过" : "未请求");
  const attempts = m.attempts || [];
  const attemptText = attempts.length
    ? attempts.map(item => {
        const usage = item.usage || {};
        const label = item.role === "format_repair" ? "修复" : (item.role === "saved" ? "复用" : "初始");
        return `${label} ${item.request_id || "无 request_id"} · prompt ${usage.prompt_tokens ?? "—"} / completion ${usage.completion_tokens ?? "—"} · ${item.finish_reason || ""}`;
      }).join("\n")
    : "没有分开保存的初始/修复响应";
  const errors = (m.errors || []).map(item => `${item.path || ""} ${item.code || ""} received=${JSON.stringify(item.received)} expected ${item.expected || ""}`).join("\n");
  const binding = m.binding || {};
  const bound = binding.object_ref ? `${binding.object_ref} · ${binding.evidence_source || ""}` : "未绑定";
  lastPlannerStage = m.stage || lastPlannerStage;
  plannerMeta.textContent = `${stage} · ${m.model || ""} · ${Number(m.elapsed_s || 0).toFixed(1)}s · 修复 ${m.repairs || 0} 次`;
  plannerBrief.textContent = [
    `输入模式: ${m.planning_input || "未知"}`,
    `真实图像: ${m.vision ? "包含" : "不包含"}`,
    `失败阶段: ${stage}`,
    `绑定对象: ${bound}`,
    binding.note || "",
    attemptText,
    errors ? "错误:\n" + errors : ""
  ].filter(Boolean).join("\n");
  plannerFull.hidden = true;
  plannerFull.textContent = JSON.stringify({attempts, errors: m.errors || [], binding}, null, 2);
  const telem = m.telemetry || {};
  if (telem.ttft_s != null) plannerBrief.textContent += `\n排队 ${telem.queue_s ?? "—"}s · 首 token ${telem.ttft_s}s · 总 ${telem.total_s}s · ${telem.endpoint || ""}`;
  if (m.stage === "binding_unresolved") renderBinding(binding);
}
function renderBinding(binding){
  const box = $("bind-box");
  const choice = $("bind-choice");
  if (!binding || binding.status === "resolved") {
    box.hidden = binding && binding.evidence_source === "sim_state" ? false : true;
    if (binding && binding.status === "resolved") {
      $("bind-note").textContent = binding.evidence_source === "sim_state"
        ? `已绑定 ${binding.object_ref}。仿真位姿辅助，不是视觉判断。`
        : `已绑定 ${binding.object_ref}（${binding.evidence_source || ""}）`;
      choice.innerHTML = "";
      $("bind-apply").disabled = true;
      box.hidden = false;
    }
    return;
  }
  box.hidden = false;
  $("bind-note").textContent = "对象未唯一绑定。请显式选择，不会默认第一个，也不会开始动作。";
  choice.innerHTML = "";
  (binding.candidates || []).forEach(ref => {
    const option = document.createElement("option");
    option.value = ref;
    option.textContent = ref;
    choice.appendChild(option);
  });
  $("bind-apply").disabled = !(binding.candidates || []).length;
}

function handleBinary(ab){
  const u8 = new Uint8Array(ab);
  const dv = new DataView(ab);
  const hlen = dv.getUint32(0);
  let hdr; try{ hdr = JSON.parse(new TextDecoder().decode(u8.subarray(4, 4+hlen))); }catch(e){ return; }
  if (gate) return;
  if (acceptEid !== null && hdr.eid && hdr.eid !== acceptEid) return;
  if (typeof hdr.scene_epoch === "number" && sceneEpoch !== null && hdr.scene_epoch < sceneEpoch) return;
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
  framesReceived += 1;
  if (frame.gap > maxFrameGapMs) maxFrameGapMs = frame.gap;
  if (hdr.primitive || hdr.frame_time || hdr.live_chunk) {
    lastFrameMeta = hdr;
    updateMotion();
  }
  if (curEid !== null && hdr.eid && hdr.eid !== curEid) clearBuffer();
  curEid = hdr.eid || curEid;
  if (!hdr.term) terminalReached = false;
  if (hdr.term) pending = frame;
  else if (!pending || !pending.hdr.term) pending = frame;
  scheduleDraw();
}
function pumpDraw(){
  if (endedFallback && !pending && !decoding && performance.now() - (endedFallback._at || (endedFallback._at = performance.now())) > 800){
    showResult(!!endedFallback.success, endedFallback.reason, endedFallback.status);
    return;
  }
  if (decoding || !pending) return;
  const f = pending;
  pending = null;
  decoding = true;
  const decodeStarted = performance.now();
  Promise.all([
    createImageBitmap(new Blob([f.aBlob], {type:"image/jpeg"})),
    createImageBitmap(new Blob([f.wBlob], {type:"image/jpeg"}))
  ]).then(([aBmp, wBmp]) => {
    const decodedAt = performance.now();
    agentCanvas.width = aBmp.width; agentCanvas.height = aBmp.height;
    wristCanvas.width = wBmp.width; wristCanvas.height = wBmp.height;
    applyAspect(agentView, f.hdr.src_aw || f.hdr.aw || aBmp.width, f.hdr.src_ah || f.hdr.ah || aBmp.height);
    applyAspect(wristView, f.hdr.src_ww || f.hdr.ww || wBmp.width, f.hdr.src_wh || f.hdr.wh || wBmp.height);
    agCtx.drawImage(aBmp, 0, 0);
    wrCtx.drawImage(wBmp, 0, 0);
    aBmp.close(); wBmp.close();
    const drawnAt = performance.now();
    if (f.hdr.frame_id != null) drawnFrameIds.add(f.hdr.frame_id);
    if (clockOffset === null && f.hdr.timestamp) clockOffset = Date.now() - f.hdr.timestamp * 1000;
    const ageMs = f.hdr.timestamp ? Math.max(0, Date.now() - f.hdr.timestamp * 1000 - (clockOffset || 0)) : 0;
    if (ageMs > maxFrameAgeMs) maxFrameAgeMs = ageMs;
    f.decodeMs = decodedAt - decodeStarted;
    f.drawMs = drawnAt - decodedAt;
    phAgent.style.display = "none"; phWrist.style.display = "none";
    lastDisplayedFid = f.hdr.fid;
    framesDrawn += 1;
    updateMotion();
    if (f.hdr.phase === "waiting_confirmation") setConfirm(true);
    if (f.hdr.term) showResult(!!f.hdr.success, f.hdr.reason, f.hdr.status || f.hdr.phase);
    else if (!terminalReached && !lastExec) setStatus(f.hdr.phase || "executing", f.hdr.detail || "");
    if (f.hdr.term || (framesDrawn % 4) === 0){
      send({
        type:"client_perf",
        gap_ms: f.gap || 0,
        decode_ms: f.decodeMs || 0,
        draw_ms: f.drawMs || 0,
        frames_received: framesReceived,
        frames_drawn: framesDrawn,
        distinct_drawn: drawnFrameIds.size,
        max_gap_ms: maxFrameGapMs,
        max_age_ms: maxFrameAgeMs
      });
    }
    decoding = false;
    if (pending) scheduleDraw();
  }).catch(() => { decoding = false; setStatus("failed", "画面解码失败", "bad"); });
}
function scheduleDraw(){
  requestAnimationFrame(pumpDraw);
}

function showResult(success, reason, status){
  terminalReached = true;
  const text = (lastExec && lastExec.reason) || reason || "";
  const kind = status || lastRunStatus;
  if (kind === "stopped_at_check" || kind === "diagnostic_saved" || kind === "stopped") {
    setStatus(kind === "diagnostic_saved" ? "预算到时" : "诊断停止", text, "warn");
  } else if (kind === "waiting_confirmation") {
    setStatus("等待确认", text, "warn");
  } else if (kind === "insufficient_evidence") {
    setStatus("证据不足", text, "warn");
  } else if (kind === "incomplete") {
    setStatus(String(text).indexOf("原生目标") >= 0 ? "原生目标已达成" : "计划已执行，环境未判定成功", text, "warn");
  } else if (success || kind === "completed") {
    setStatus(String(text).indexOf("用户目标") >= 0 ? "用户目标完成" : "任务完成", text, "good");
  } else if (lastPlannerStage === "ok") {
    setStatus("规划通过，执行失败", text, "bad");
  } else {
    setStatus("执行失败", reason || "", "bad");
  }
  restartBtn.disabled = false;
  syncRunButtons();
  setConfirm(false);
  endedFallback = null;
}

function applyAspect(view, width, height){
  if (width > 0 && height > 0) view.style.setProperty("--ar", `${width} / ${height}`);
}

function tick(){
  requestAnimationFrame(tick);
  pumpDraw();
}

loadBtn.onclick = () => {
  if (!wsConnected) return;
  if (!suiteSel.value) return;
  loadBtn.disabled = true;
  setStatus("加载中", suiteSel.value);
  send({type:"load", suite: suiteSel.value, client_request_id: `load-${Date.now()}`});
};
function applySnapshot(m){
  lastSnapshot = m || null;
  const scene = m && m.scene;
  if (scene && scene.loaded) {
    episode = scene.episode_id;
    sceneEpoch = scene.scene_epoch;
    if (scene.plan_version) planVersion = scene.plan_version;
    if (scene.user_text) goalEl.textContent = `输入保留：${scene.user_text}`;
    else if (!taskEdited && scene.summary_zh) goalEl.textContent = `当前目标：${scene.summary_zh}`;
    taskIn.disabled = false;
  }
  const run = (m && m.run) || {};
  rootError = run.root_error || "";
  if (run.phase === "waiting_confirmation") setConfirm(true);
  else if (run.phase === "succeeded" || run.phase === "failed" || run.phase === "cancelled" || run.phase === "idle") setConfirm(false);
  if (run.execution) rememberExec(run.execution);
  if (run.user_text) goalEl.textContent = `输入保留：${run.user_text}` + (run.plan_summary ? ` · 计划：${run.plan_summary}` : "");
  if (progressEl && run.phase && run.phase !== "idle") {
    const nativeText = run.native_task_success == null ? "" : (run.native_task_success ? "原生目标已达成" : "原生目标未达成");
    const userText = run.user_goal_success == null ? "" : (run.user_goal_success ? "用户目标已完成" : "用户目标未完成");
    const bits = [run.execution_mode === "free" ? "自由模式" : "", run.phase, nativeText, userText].filter(Boolean);
    progressEl.textContent = bits.join(" · ");
  }
  if (rootError && (run.phase === "failed" || run.phase === "cancelled" || run.phase === "waiting_confirmation")) {
    setStatus(run.phase === "waiting_confirmation" ? "等待确认" : (run.phase === "cancelled" ? "已取消" : "失败"), rootError, run.phase === "waiting_confirmation" ? "warn" : "bad");
  }
  syncRunButtons();
  scheduleDraw();
}
function applyServerScene(m){
  const scene = m && m.scene;
  if (!scene || !scene.loaded) {
    if (m && (m.loading || m.env === "loading" || (m.can_load === false && m.block_code && m.block_code !== "scene_missing"))) return;
    episode = null;
    sceneEpoch = null;
    return;
  }
  episode = scene.episode_id;
  sceneEpoch = scene.scene_epoch;
  if (scene.plan_version) planVersion = scene.plan_version;
  if (!taskEdited && scene.summary_zh) goalEl.textContent = `当前目标：${scene.summary_zh}`;
  taskIn.disabled = !!m.running;
}
function begin(source){
  if (!ws || ws.readyState !== 1) {
    setStatus("失败", "连接未就绪", "bad");
    return;
  }
  if (sceneEpoch === null || sceneEpoch === undefined) {
    setStatus("已拒绝", "场景未加载", "bad");
    return;
  }
  const type = source === "restart" ? "restart" : "start";
  const client_request_id = `req-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  send({
    type,
    task: taskIn.value,
    plan_source: "model",
    planner_profile: profileSel.value,
    client_request_id,
    scene_epoch: sceneEpoch,
    execution_mode: modeSel ? modeSel.value : "free"
  });
  setStatus(source === "restart" ? "重置" : "检查任务", client_request_id);
}
startBtn.onclick = () => begin("model");
restartBtn.onclick = () => begin("restart");
stopBtn.onclick = () => { send({type:"stop", client_request_id: `stop-${Date.now()}`}); };
profileSel.onchange = () => send({type:"set_planner", profile: profileSel.value});
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
$("exec-expand").onclick = () => { execFull.hidden = !execFull.hidden; };
taskIn.addEventListener("input", ()=>{
  taskEdited = true;
  syncRunButtons();
});
$("bind-apply").onclick = () => {
  const ref = $("bind-choice").value;
  if (!ref) return;
  send({type: "bind", object_ref: ref});
};
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
setInterval(async () => {
  try {
    const response = await fetch("/health", {cache: "no-store"});
    if (response.ok) renderHealth(await response.json());
  } catch (err) {}
}, 5000);
connect();
bootstrap();
requestAnimationFrame(tick);
