const video = document.querySelector("#camera");
const placeholder = document.querySelector("#camera-placeholder");

const startButton = document.querySelector("#start-camera");
const switchButton = document.querySelector("#switch-camera");
const stopButton = document.querySelector("#stop-camera");

const statusElement = document.querySelector("#status");
const protocolStatus = document.querySelector("#protocol-status");
const secureStatus = document.querySelector("#secure-status");
const mediaStatus = document.querySelector("#media-status");
const cameraStatus = document.querySelector("#camera-status");
const facingStatus = document.querySelector("#facing-status");
const resolutionStatus = document.querySelector("#resolution-status");
const errorStatus = document.querySelector("#error-status");

let stream = null;
let facingMode = "user";

/* ----------------------------------
   Debug helpers
---------------------------------- */

function setStatus(message) {
  statusElement.textContent = message;

  console.log(`[Human AR] ${message}`);
}

function setError(error) {
  console.error("[Human AR Camera Error]", error);

  errorStatus.textContent =
    `${error.name}: ${error.message}`;
}

/* ----------------------------------
   Browser capability
---------------------------------- */

function checkEnvironment() {
  protocolStatus.textContent =
    window.location.protocol;

  secureStatus.textContent =
    window.isSecureContext ? "YES" : "NO";

  const supported = !!(
    navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia
  );

  mediaStatus.textContent =
    supported ? "Supported" : "Not Supported";

  if (!supported) {
    setStatus("Camera API unavailable");

    errorStatus.textContent =
      "navigator.mediaDevices.getUserMedia unavailable";

    startButton.disabled = true;

    return;
  }

  setStatus("Environment ready");
}

/* ----------------------------------
   Camera display orientation
---------------------------------- */

function updateCameraDisplay() {

  /*
    Front camera:
    mirror the PREVIEW so it behaves
    like a normal phone selfie camera.

    Rear camera:
    show the real orientation.
  */

  if (facingMode === "user") {

    video.style.transform = "scaleX(-1)";

  } else {

    video.style.transform = "none";

  }

}

/* ----------------------------------
   Stop stream
---------------------------------- */

function stopStream() {

  if (!stream) {
    return;
  }

  for (const track of stream.getTracks()) {
    track.stop();
  }

  stream = null;

  video.srcObject = null;

  video.style.display = "none";

  placeholder.style.display = "flex";
  placeholder.textContent = "Camera stopped";

  cameraStatus.textContent = "Stopped";

  resolutionStatus.textContent = "—";

  switchButton.disabled = true;
  stopButton.disabled = true;
  startButton.disabled = false;
}

/* ----------------------------------
   Start camera
---------------------------------- */

async function startCamera() {

  errorStatus.textContent = "None";

  setStatus("Requesting camera...");

  startButton.disabled = true;

  try {

    if (stream) {
      stopStream();
    }

    const constraints = {

      audio: false,

      video: {

        facingMode: {
          ideal: facingMode
        },

        width: {
          ideal: 1280
        },

        height: {
          ideal: 720
        }

      }

    };

    stream =
      await navigator.mediaDevices.getUserMedia(
        constraints
      );

    video.srcObject = stream;

    await video.play();

    /*
      Apply display orientation only
      after the camera has started.
    */

    updateCameraDisplay();

    video.style.display = "block";

    placeholder.style.display = "none";

    cameraStatus.textContent = "Running";

    facingStatus.textContent = facingMode;

    updateResolution();

    startButton.disabled = true;
    switchButton.disabled = false;
    stopButton.disabled = false;

    setStatus("Camera running");

  } catch (error) {

    stream = null;

    setError(error);

    setStatus("Camera failed");

    video.style.display = "none";

    placeholder.style.display = "flex";
    placeholder.textContent = "Camera unavailable";

    cameraStatus.textContent = "Failed";

    startButton.disabled = false;
    switchButton.disabled = true;
    stopButton.disabled = true;

  }

}

/* ----------------------------------
   Resolution
---------------------------------- */

function updateResolution() {

  const width = video.videoWidth;
  const height = video.videoHeight;

  if (width && height) {

    resolutionStatus.textContent =
      `${width} × ${height}`;

  } else {

    resolutionStatus.textContent =
      "Waiting...";

  }

}

/* ----------------------------------
   Switch front / rear
---------------------------------- */

async function switchCamera() {

  facingMode =
    facingMode === "user"
      ? "environment"
      : "user";

  facingStatus.textContent = facingMode;

  setStatus(
    `Switching to ${facingMode} camera...`
  );

  stopStream();

  await startCamera();

}

/* ----------------------------------
   Stop camera
---------------------------------- */

function stopCamera() {

  stopStream();

  setStatus("Camera stopped");

}

/* ----------------------------------
   Events
---------------------------------- */

startButton.addEventListener(
  "click",
  startCamera
);

switchButton.addEventListener(
  "click",
  switchCamera
);

stopButton.addEventListener(
  "click",
  stopCamera
);

video.addEventListener(
  "loadedmetadata",
  updateResolution
);

/* ----------------------------------
   Initialization
---------------------------------- */

console.log(
  "[Human AR] Milestone 2 initialized"
);

checkEnvironment();