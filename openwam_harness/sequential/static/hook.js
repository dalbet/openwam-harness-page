(function () {
  const Native = WebSocket;
  function Wrapped(url, protocols) {
    const ws = protocols === undefined ? new Native(url) : new Native(url, protocols);
    window.__pageWs = ws;
    return ws;
  }
  Wrapped.prototype = Native.prototype;
  window.WebSocket = Wrapped;

  function send(obj) {
    const ws = window.__pageWs;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
  }

  function bind() {
    const stop = document.getElementById("stop");
    const ok = document.getElementById("ok");
    const bad = document.getElementById("bad");
    const unk = document.getElementById("unk");
    const start = document.getElementById("start");
    const restart = document.getElementById("restart");
    function arm(on) {
      stop.disabled = !on;
      ok.disabled = !on;
      bad.disabled = !on;
      unk.disabled = !on;
    }
    start.addEventListener("click", () => arm(true));
    restart.addEventListener("click", () => arm(true));
    stop.addEventListener("click", () => { send({type: "stop"}); arm(false); });
    ok.addEventListener("click", () => send({type: "confirm", verdict: "success"}));
    bad.addEventListener("click", () => send({type: "confirm", verdict: "failure"}));
    unk.addEventListener("click", () => send({type: "confirm", verdict: "unknown"}));
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
