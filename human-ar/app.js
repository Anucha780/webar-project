import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

import * as THREE
  from "https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js";

/* =========================================================
   DOM
========================================================= */

const video =
  document.querySelector("#camera");

const threeLayer =
  document.querySelector("#three-layer");

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

const threeStatus =
  document.querySelector("#three-status");

const anchorStatus =
  document.querySelector("#anchor-status");

const leftShoulderStatus =
  document.querySelector("#left-shoulder-status");

const rightShoulderStatus =
  document.querySelector("#right-shoulder-status");

const errorStatus =
  document.querySelector("#error-status");


/* =========================================================
   STATE
========================================================= */

let stream = null;

let facingMode = "user";

let poseLandmarker = null;

let cameraRunning = false;

let animationFrameId = null;

let lastVideoTime = -1;

let frameCounter = 0;

let fpsTimer = performance.now();


/* =========================================================
   THREE.JS STATE
========================================================= */

let scene = null;

let threeCamera = null;

let renderer = null;

let sphere = null;


/* =========================================================
   LANDMARK INDEXES
========================================================= */

const LANDMARK = {
  NOSE: 0,

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_HIP: 23,
  RIGHT_HIP: 24
};


/* =========================================================
   DEBUG
========================================================= */

function setStatus(message) {
  statusElement.textContent =
    message;

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


/* =========================================================
   THREE.JS INITIALIZATION
========================================================= */

function initializeThree() {

  try {

    threeStatus.textContent =
      "Initializing...";

    scene =
      new THREE.Scene();


    /*
      Orthographic camera is intentional.

      M4 is screen-space body tracking.

      Using an orthographic camera makes
      screen coordinate → Three.js coordinate
      mapping deterministic.
    */

    threeCamera =
      new THREE.OrthographicCamera(
        0,
        1,
        1,
        0,
        -100,
        100
      );

    threeCamera.position.z =
      10;


    renderer =
      new THREE.WebGLRenderer({
        alpha: true,
        antialias: true
      });

    renderer.setClearColor(
      0x000000,
      0
    );

    renderer.setPixelRatio(
      Math.min(
        window.devicePixelRatio || 1,
        2
      )
    );

    threeLayer.appendChild(
      renderer.domElement
    );


    /*
      Test primitive.

      We deliberately use a basic sphere
      before introducing GLB.
    */

    const geometry =
      new THREE.SphereGeometry(
        0.045,
        32,
        24
      );

    const material =
      new THREE.MeshNormalMaterial();

    sphere =
      new THREE.Mesh(
        geometry,
        material
      );

    sphere.visible =
      false;

    sphere.position.z =
      0;

    scene.add(
      sphere
    );

    resizeThree();

    renderer.render(
      scene,
      threeCamera
    );

    threeStatus.textContent =
      `Ready r${THREE.REVISION}`;

    console.log(
      `[Human AR] Three.js revision ${THREE.REVISION}`
    );

  } catch (error) {

    threeStatus.textContent =
      "Failed";

    setError(error);

    throw error;
  }
}


/* =========================================================
   THREE.JS RESIZE
========================================================= */

function resizeThree() {

  if (
    !renderer ||
    !threeCamera
  ) {
    return;
  }

  const rect =
    threeLayer.getBoundingClientRect();

  if (
    rect.width <= 0 ||
    rect.height <= 0
  ) {
    return;
  }

  renderer.setSize(
    rect.width,
    rect.height,
    false
  );


  /*
    Coordinate system:

    X: 0 → 1
    Y: 0 → 1

    This matches normalized screen-space
    coordinates.
  */

  threeCamera.left = 0;
  threeCamera.right = 1;

  threeCamera.top = 1;
  threeCamera.bottom = 0;

  threeCamera.updateProjectionMatrix();
}


/* =========================================================
   MEDIAPIPE
========================================================= */

async function initializePose() {

  try {

    setStatus(
      "Loading MediaPipe Pose..."
    );

    mediapipeStatus.textContent =
      "Loading WASM...";

    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
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
      "Pose + Three.js ready — start camera"
    );

    startButton.disabled =
      false;

  } catch (error) {

    mediapipeStatus.textContent =
      "Failed";

    setError(error);

    setStatus(
      "MediaPipe failed"
    );

    startButton.disabled =
      true;
  }
}


/* =========================================================
   CAMERA DISPLAY
========================================================= */

function updateCameraDisplay() {

  if (
    facingMode === "user"
  ) {

    video.style.transform =
      "scaleX(-1)";

  } else {

    video.style.transform =
      "none";
  }
}


/* =========================================================
   START CAMERA
========================================================= */

async function startCamera() {

  if (
    !poseLandmarker ||
    !renderer
  ) {

    setStatus(
      "Tracking system is not ready"
    );

    return;
  }

  errorStatus.textContent =
    "None";

  setStatus(
    "Requesting camera..."
  );

  startButton.disabled =
    true;

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

    threeLayer.style.display =
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

    resizeThree();

    startButton.disabled =
      true;

    switchButton.disabled =
      false;

    stopButton.disabled =
      false;

    lastVideoTime =
      -1;

    setStatus(
      "Camera + Pose + Three.js running"
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

    threeLayer.style.display =
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


/* =========================================================
   STOP
========================================================= */

function stopStream() {

  cameraRunning =
    false;

  if (
    animationFrameId !== null
  ) {

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

  threeLayer.style.display =
    "none";

  overlay.style.display =
    "none";

  if (sphere) {
    sphere.visible = false;
  }

  anchorStatus.textContent =
    "Hidden";

  clearOverlay();

  if (
    renderer &&
    scene &&
    threeCamera
  ) {

    renderer.render(
      scene,
      threeCamera
    );
  }

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


/* =========================================================
   SWITCH CAMERA
========================================================= */

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


/* =========================================================
   SIZE
========================================================= */

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


/* =========================================================
   OBJECT-FIT COVER TRANSFORM
========================================================= */

function getCoverTransform() {

  const sourceWidth =
    video.videoWidth;

  const sourceHeight =
    video.videoHeight;

  const displayWidth =
    overlay.width;

  const displayHeight =
    overlay.height;

  if (
    !sourceWidth ||
    !sourceHeight ||
    !displayWidth ||
    !displayHeight
  ) {

    return null;
  }

  const scale =
    Math.max(
      displayWidth /
        sourceWidth,

      displayHeight /
        sourceHeight
    );

  const renderedWidth =
    sourceWidth * scale;

  const renderedHeight =
    sourceHeight * scale;

  const cropX =
    (
      renderedWidth -
      displayWidth
    ) / 2;

  const cropY =
    (
      renderedHeight -
      displayHeight
    ) / 2;

  return {
    sourceWidth,
    sourceHeight,

    displayWidth,
    displayHeight,

    scale,

    cropX,
    cropY
  };
}


/* =========================================================
   MEDIAPIPE → CANVAS
========================================================= */

function landmarkToCanvas(
  landmark
) {

  const transform =
    getCoverTransform();

  if (!transform) {

    return {
      x: 0,
      y: 0
    };
  }

  let sourceX =
    landmark.x *
    transform.sourceWidth;

  const sourceY =
    landmark.y *
    transform.sourceHeight;

  if (
    facingMode === "user"
  ) {

    sourceX =
      transform.sourceWidth -
      sourceX;
  }

  const x =
    sourceX *
    transform.scale -
    transform.cropX;

  const y =
    sourceY *
    transform.scale -
    transform.cropY;

  return {
    x,
    y
  };
}


/* =========================================================
   CANVAS → THREE SCREEN SPACE
========================================================= */

function canvasToThree(
  point
) {

  /*
    Canvas:
      x = 0 → width
      y = 0 → height

    Three orthographic scene:
      x = 0 → 1
      y = 0 → 1

    Canvas Y grows downward.
    Three Y grows upward.
  */

  return {

    x:
      point.x /
      overlay.width,

    y:
      1 -
      (
        point.y /
        overlay.height
      )
  };
}


/* =========================================================
   UPDATE 3D SPHERE
========================================================= */

function updateThreeAnchor(
  landmarks
) {

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
    Synthetic torso center.
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


  /*
    Use exactly the SAME mapping as
    the debug overlay.
  */

  const canvasPoint =
    landmarkToCanvas(
      torso
    );

  const threePoint =
    canvasToThree(
      canvasPoint
    );


  /*
    Do not show an object that has moved
    completely outside the visible viewport.
  */

  const visible =
    threePoint.x >= 0 &&
    threePoint.x <= 1 &&
    threePoint.y >= 0 &&
    threePoint.y <= 1;

  sphere.visible =
    visible;

  if (!visible) {

    anchorStatus.textContent =
      "Outside view";

    return;
  }


  sphere.position.set(
    threePoint.x,
    threePoint.y,
    0
  );


  /*
    Rotate it slowly so we can visually
    confirm that this is a real Three.js
    3D mesh, not a 2D circle.
  */

  sphere.rotation.x +=
    0.025;

  sphere.rotation.y +=
    0.035;

  anchorStatus.textContent =
    `TORSO x ${threePoint.x.toFixed(3)}, y ${threePoint.y.toFixed(3)}`;
}


/* =========================================================
   DEBUG DRAWING
========================================================= */

function drawPoint(
  landmark,
  label
) {

  const point =
    landmarkToCanvas(
      landmark
    );

  const dpr =
    window.devicePixelRatio || 1;

  const radius =
    6 * dpr;

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
    `${11 * dpr}px Arial`;

  ctx.fillStyle =
    "#ffffff";

  ctx.fillText(
    label,
    point.x +
      radius +
      3 * dpr,

    point.y -
      3 * dpr
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

  const dpr =
    window.devicePixelRatio || 1;

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
    2 * dpr;

  ctx.stroke();
}


/* =========================================================
   DRAW POSE
========================================================= */

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

  leftShoulderStatus.textContent =
    `x ${leftShoulder.x.toFixed(3)}, y ${leftShoulder.y.toFixed(3)}`;

  rightShoulderStatus.textContent =
    `x ${rightShoulder.x.toFixed(3)}, y ${rightShoulder.y.toFixed(3)}`;
}


/* =========================================================
   FPS
========================================================= */

function updateFPS() {

  frameCounter++;

  const now =
    performance.now();

  const elapsed =
    now -
    fpsTimer;

  if (
    elapsed >= 1000
  ) {

    const fps =
      Math.round(
        (
          frameCounter *
          1000
        ) /
        elapsed
      );

    fpsStatus.textContent =
      `${fps}`;

    frameCounter =
      0;

    fpsTimer =
      now;
  }
}


/* =========================================================
   POSE LOOP
========================================================= */

function startPoseLoop() {

  if (
    !cameraRunning
  ) {
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

        const landmarks =
          result.landmarks[0];

        drawPose(
          landmarks
        );

        updateThreeAnchor(
          landmarks
        );

      } else {

        personStatus.textContent =
          "Not detected";

        leftShoulderStatus.textContent =
          "—";

        rightShoulderStatus.textContent =
          "—";

        clearOverlay();

        if (sphere) {
          sphere.visible = false;
        }

        anchorStatus.textContent =
          "Hidden";
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


  /*
    Render Three.js every animation frame.
  */

  if (
    renderer &&
    scene &&
    threeCamera
  ) {

    renderer.render(
      scene,
      threeCamera
    );
  }


  animationFrameId =
    requestAnimationFrame(
      predictPose
    );
}


/* =========================================================
   EVENTS
========================================================= */

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

    resizeThree();
  }
);


window.addEventListener(
  "resize",
  () => {

    if (
      cameraRunning
    ) {

      resizeOverlay();

      resizeThree();
    }
  }
);


/* =========================================================
   INITIALIZATION
========================================================= */

console.log(
  "[Human AR] Milestone 4 initialized"
);

startButton.disabled =
  true;


/*
  Initialize Three.js synchronously first.

  If this fails, we do NOT proceed silently.
*/

try {

  initializeThree();

} catch (error) {

  setStatus(
    "Three.js failed"
  );

  startButton.disabled =
    true;
}


/*
  MediaPipe initialization remains independent.
*/

initializePose();