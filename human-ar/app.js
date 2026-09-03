import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

/* ----------------------------------
   DOM
---------------------------------- */

const video =
  document.querySelector("#camera");

const overlay =
  document.querySelector("#pose-overlay");

const ctx =
  overlay.getContext("2d");

const placeholder =
  document.querySelector("#camera-placeholder");

const startButton =
  document.querySelector("#start-camera");

const switchButton =
  document.querySelector("#switch-camera");

const stopButton =
  document.querySelector("#stop-camera");

const statusElement =
  document.querySelector("#status");

const cameraStatus =
  document.querySelector("#camera-status");

const facingStatus =
  document.querySelector("#facing-status");

const resolutionStatus =
  document.querySelector("#resolution-status");

const mediapipeStatus =
  document.querySelector("#mediapipe-status");

const personStatus =
  document.querySelector("#person-status");

const fpsStatus =
  document.querySelector("#fps-status");

const leftShoulderStatus =
  document.querySelector("#left-shoulder-status");

const rightShoulderStatus =
  document.querySelector("#right-shoulder-status");

const errorStatus =
  document.querySelector("#error-status");


/* ----------------------------------
   State
---------------------------------- */

let stream = null;

let facingMode = "user";

let poseLandmarker = null;

let cameraRunning = false;

let animationFrameId = null;

let lastVideoTime = -1;

let frameCounter = 0;

let fpsTimer = performance.now();


/* ----------------------------------
   Important landmark indexes
---------------------------------- */

/*
  MediaPipe Pose landmark indexes:

  0  = nose

  11 = left shoulder
  12 = right shoulder

  23 = left hip
  24 = right hip
*/

const LANDMARK = {
  NOSE: 0,

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_HIP: 23,
  RIGHT_HIP: 24
};


/* ----------------------------------
   Debug
---------------------------------- */

function setStatus(message) {

  statusElement.textContent = message;

  console.log(
    `[Human AR] ${message}`
  );

}

function setError(error) {

  console.error(
    "[Human AR Error]",
    error
  );

  errorStatus.textContent =
    `${error.name}: ${error.message}`;

}


/* ----------------------------------
   MediaPipe initialization
---------------------------------- */

async function initializePose() {

  try {

    setStatus(
      "Loading MediaPipe Pose..."
    );

    mediapipeStatus.textContent =
      "Loading WASM...";

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
      );

    mediapipeStatus.textContent =
      "Loading model...";

    poseLandmarker =
      await PoseLandmarker.createFromOptions(
        vision,
        {

          baseOptions: {

            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",

            delegate: "GPU"

          },

          runningMode: "VIDEO",

          numPoses: 1,

          minPoseDetectionConfidence: 0.5,

          minPosePresenceConfidence: 0.5,

          minTrackingConfidence: 0.5

        }
      );

    mediapipeStatus.textContent =
      "Ready";

    setStatus(
      "Pose ready — start camera"
    );

    startButton.disabled = false;

  } catch (error) {

    mediapipeStatus.textContent =
      "Failed";

    setError(error);

    setStatus(
      "MediaPipe failed"
    );

    startButton.disabled = true;

  }

}


/* ----------------------------------
   Camera display
---------------------------------- */

function updateCameraDisplay() {

  /*
    Only the visible preview is
    mirrored for the front camera.

    MediaPipe still receives the
    original raw video.
  */

  if (facingMode === "user") {

    video.style.transform =
      "scaleX(-1)";

  } else {

    video.style.transform =
      "none";

  }

}


/* ----------------------------------
   Camera
---------------------------------- */

async function startCamera() {

  if (!poseLandmarker) {

    setStatus(
      "Pose model is not ready"
    );

    return;

  }

  errorStatus.textContent =
    "None";

  setStatus(
    "Requesting camera..."
  );

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
      await navigator.mediaDevices
        .getUserMedia(
          constraints
        );

    video.srcObject =
      stream;

    await video.play();

    updateCameraDisplay();

    video.style.display =
      "block";

    overlay.style.display =
      "block";

    placeholder.style.display =
      "none";

    cameraRunning =
      true;

    cameraStatus.textContent =
      "Running";

    facingStatus.textContent =
      facingMode;

    updateResolution();

    resizeOverlay();

    startButton.disabled =
      true;

    switchButton.disabled =
      false;

    stopButton.disabled =
      false;

    lastVideoTime = -1;

    setStatus(
      "Camera + Pose running"
    );

    startPoseLoop();

  } catch (error) {

    cameraRunning =
      false;

    stream =
      null;

    setError(error);

    setStatus(
      "Camera failed"
    );

    video.style.display =
      "none";

    overlay.style.display =
      "none";

    placeholder.style.display =
      "flex";

    placeholder.textContent =
      "Camera unavailable";

    cameraStatus.textContent =
      "Failed";

    startButton.disabled =
      false;

    switchButton.disabled =
      true;

    stopButton.disabled =
      true;

  }

}


function stopStream() {

  cameraRunning =
    false;

  if (animationFrameId !== null) {

    cancelAnimationFrame(
      animationFrameId
    );

    animationFrameId =
      null;

  }

  if (stream) {

    for (
      const track
      of stream.getTracks()
    ) {

      track.stop();

    }

  }

  stream =
    null;

  video.srcObject =
    null;

  video.style.display =
    "none";

  overlay.style.display =
    "none";

  clearOverlay();

  placeholder.style.display =
    "flex";

  placeholder.textContent =
    "Camera stopped";

  cameraStatus.textContent =
    "Stopped";

  personStatus.textContent =
    "Not detected";

  fpsStatus.textContent =
    "—";

  leftShoulderStatus.textContent =
    "—";

  rightShoulderStatus.textContent =
    "—";

  resolutionStatus.textContent =
    "—";

  switchButton.disabled =
    true;

  stopButton.disabled =
    true;

  startButton.disabled =
    !poseLandmarker;

}


function stopCamera() {

  stopStream();

  setStatus(
    "Camera stopped"
  );

}


async function switchCamera() {

  facingMode =
    facingMode === "user"
      ? "environment"
      : "user";

  facingStatus.textContent =
    facingMode;

  setStatus(
    `Switching to ${facingMode} camera...`
  );

  stopStream();

  await startCamera();

}


/* ----------------------------------
   Resolution / canvas
---------------------------------- */

function updateResolution() {

  if (
    video.videoWidth &&
    video.videoHeight
  ) {

    resolutionStatus.textContent =
      `${video.videoWidth} × ${video.videoHeight}`;

  } else {

    resolutionStatus.textContent =
      "Waiting...";

  }

}


function resizeOverlay() {

  const rect =
    overlay.getBoundingClientRect();

  const dpr =
    window.devicePixelRatio || 1;

  overlay.width =
    Math.round(
      rect.width * dpr
    );

  overlay.height =
    Math.round(
      rect.height * dpr
    );

}


function clearOverlay() {

  ctx.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );

}


/* ----------------------------------
   Coordinate mapping
---------------------------------- */

function landmarkToCanvas(
  landmark
) {

  /*
    IMPORTANT:

    MediaPipe receives the RAW frame.

    Front camera preview is mirrored.

    Therefore overlay X must also
    be mirrored for the front camera.
  */

  let normalizedX =
    landmark.x;

  if (
    facingMode === "user"
  ) {

    normalizedX =
      1 - normalizedX;

  }

  return {

    x:
      normalizedX *
      overlay.width,

    y:
      landmark.y *
      overlay.height

  };

}


/* ----------------------------------
   Drawing
---------------------------------- */

function drawPoint(
  landmark,
  label
) {

  const point =
    landmarkToCanvas(
      landmark
    );

  const radius =
    7 *
    (window.devicePixelRatio || 1);

  ctx.beginPath();

  ctx.arc(
    point.x,
    point.y,
    radius,
    0,
    Math.PI * 2
  );

  ctx.fillStyle =
    "#00ff88";

  ctx.fill();

  ctx.font =
    `${12 * (window.devicePixelRatio || 1)}px Arial`;

  ctx.fillStyle =
    "#ffffff";

  ctx.fillText(
    label,
    point.x + radius + 4,
    point.y
  );

}


function drawLine(
  landmarkA,
  landmarkB
) {

  const a =
    landmarkToCanvas(
      landmarkA
    );

  const b =
    landmarkToCanvas(
      landmarkB
    );

  ctx.beginPath();

  ctx.moveTo(
    a.x,
    a.y
  );

  ctx.lineTo(
    b.x,
    b.y
  );

  ctx.strokeStyle =
    "#00ff88";

  ctx.lineWidth =
    3 *
    (window.devicePixelRatio || 1);

  ctx.stroke();

}


function drawPose(
  landmarks
) {

  clearOverlay();

  const nose =
    landmarks[
      LANDMARK.NOSE
    ];

  const leftShoulder =
    landmarks[
      LANDMARK.LEFT_SHOULDER
    ];

  const rightShoulder =
    landmarks[
      LANDMARK.RIGHT_SHOULDER
    ];

  const leftHip =
    landmarks[
      LANDMARK.LEFT_HIP
    ];

  const rightHip =
    landmarks[
      LANDMARK.RIGHT_HIP
    ];


  /*
    Torso skeleton
  */

  drawLine(
    leftShoulder,
    rightShoulder
  );

  drawLine(
    leftShoulder,
    leftHip
  );

  drawLine(
    rightShoulder,
    rightHip
  );

  drawLine(
    leftHip,
    rightHip
  );


  /*
    Important anchor points
  */

  drawPoint(
    nose,
    "HEAD"
  );

  drawPoint(
    leftShoulder,
    "L SHOULDER"
  );

  drawPoint(
    rightShoulder,
    "R SHOULDER"
  );

  drawPoint(
    leftHip,
    "L HIP"
  );

  drawPoint(
    rightHip,
    "R HIP"
  );


  /*
    Torso center
  */

  const torso = {

    x:
      (
        leftShoulder.x +
        rightShoulder.x +
        leftHip.x +
        rightHip.x
      ) / 4,

    y:
      (
        leftShoulder.y +
        rightShoulder.y +
        leftHip.y +
        rightHip.y
      ) / 4

  };

  drawPoint(
    torso,
    "TORSO"
  );


  /*
    Debug coordinates
  */

  leftShoulderStatus.textContent =
    `x ${leftShoulder.x.toFixed(3)}, y ${leftShoulder.y.toFixed(3)}`;

  rightShoulderStatus.textContent =
    `x ${rightShoulder.x.toFixed(3)}, y ${rightShoulder.y.toFixed(3)}`;

}


/* ----------------------------------
   FPS
---------------------------------- */

function updateFPS() {

  frameCounter++;

  const now =
    performance.now();

  const elapsed =
    now - fpsTimer;

  if (
    elapsed >= 1000
  ) {

    const fps =
      Math.round(
        (
          frameCounter *
          1000
        ) / elapsed
      );

    fpsStatus.textContent =
      `${fps}`;

    frameCounter =
      0;

    fpsTimer =
      now;

  }

}


/* ----------------------------------
   Pose loop
---------------------------------- */

function startPoseLoop() {

  if (!cameraRunning) {
    return;
  }

  predictPose();

}


function predictPose() {

  if (
    !cameraRunning ||
    !poseLandmarker
  ) {

    return;

  }

  if (
    video.readyState >= 2 &&
    video.currentTime !==
      lastVideoTime
  ) {

    lastVideoTime =
      video.currentTime;

    try {

      const result =
        poseLandmarker
          .detectForVideo(
            video,
            performance.now()
          );

      if (
        result.landmarks &&
        result.landmarks.length > 0
      ) {

        personStatus.textContent =
          "Detected";

        drawPose(
          result.landmarks[0]
        );

      } else {

        personStatus.textContent =
          "Not detected";

        leftShoulderStatus.textContent =
          "—";

        rightShoulderStatus.textContent =
          "—";

        clearOverlay();

      }

      updateFPS();

    } catch (error) {

      setError(error);

      setStatus(
        "Pose detection error"
      );

      cameraRunning =
        false;

      return;

    }

  }

  animationFrameId =
    requestAnimationFrame(
      predictPose
    );

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
  () => {

    updateResolution();

    resizeOverlay();

  }
);

window.addEventListener(
  "resize",
  () => {

    if (
      cameraRunning
    ) {

      resizeOverlay();

    }

  }
);


/* ----------------------------------
   Start M3
---------------------------------- */

console.log(
  "[Human AR] Milestone 3 initialized"
);

/*
  Start button remains disabled
  until MediaPipe is ready.
*/

startButton.disabled =
  true;

initializePose();