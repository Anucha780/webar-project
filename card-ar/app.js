const statusElement = document.querySelector("#status");
const jsStatusElement = document.querySelector("#js-status");
const protocolStatusElement =
  document.querySelector("#protocol-status");

function initialize() {
  console.log("[Card AR] Milestone 1 starting");

  jsStatusElement.textContent = "ES Module OK";

  protocolStatusElement.textContent =
    window.location.protocol;

  statusElement.textContent = "Milestone 1 Ready";

  console.log("[Card AR] Milestone 1 ready");
}

initialize();