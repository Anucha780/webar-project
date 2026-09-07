console.log("APP.JS STARTED");

import {
  FilesetResolver,
  PoseLandmarker,
  ImageSegmenter
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

import * as THREE from "three";

import {
  GLTFLoader
} from "three/addons/loaders/GLTFLoader.js";

import {
  MODEL_REGISTRY,
  DEFAULT_MODEL_ID,
  getModelConfig,
  normalizeModelConfig
} from "./models.js";


/* =========================================================
   DOM
========================================================= */

const video =
  document.querySelector("#camera");

const threeLayer =
  document.querySelector("#three-layer");

const frontThreeLayer =
  document.querySelector(
    "#three-front-layer"
  );

const overlay =
  document.querySelector("#pose-overlay");

const effectOverlay =
  document.querySelector("#effect-overlay");

const ctx =
  overlay.getContext(
    "2d"
  );

const effectCtx =
  effectOverlay.getContext(
    "2d"
  );

const startButton =
  document.querySelector(
    "#start-camera"
  );

const stopButton =
  document.querySelector(
    "#stop-camera"
  );

const switchButton =
  document.querySelector(
    "#switch-camera"
  );

const captureButton =
  document.querySelector(
    "#capture-photo"
  );

const toggleStars =
  document.querySelector(
    "#toggle-stars"
  );

const modelToggleContainer =
  document.querySelector(
    "#model-toggle-container"
  );

const modelToggleElements =
  new Map();


/* =========================================================
   DEBUG UI
========================================================= */

const cameraStatus =
  document.querySelector(
    "#camera-status"
  );

const poseStatus =
  document.querySelector(
    "#pose-status"
  );

const segmentationStatus =
  document.querySelector(
    "#segmentation-status"
  );

const threeStatus =
  document.querySelector(
    "#three-status"
  );

const modelStatus =
  document.querySelector(
    "#model-status"
  );

const anchorStatus =
  document.querySelector(
    "#anchor-status"
  );

const effectStatus =
  document.querySelector(
    "#effect-status"
  );

const captureStatus =
  document.querySelector(
    "#capture-status"
  );

const errorStatus =
  document.querySelector(
    "#error-status"
  );


/* =========================================================
   CAMERA STATE
========================================================= */

let stream = null;

let cameraRunning =
  false;

let facingMode =
  "user";

let animationFrameId =
  null;

let lastFrameTimestamp =
  performance.now();

let lastVideoTime =
  -1;


/* =========================================================
   MEDIAPIPE STATE
========================================================= */

let poseLandmarker =
  null;

let imageSegmenter =
  null;

let latestLandmarks =
  null;

let latestSegmentationMask =
  null;

let latestSegmentationTimestamp =
  0;


/* =========================================================
   THREE STATE
========================================================= */

let scene = null;

let threeCamera = null;

let renderer = null;

let frontRenderer = null;

const modelInstances =
  new Map();

let allModelsReady =
  false;


/* =========================================================
   ORBIT STATE
========================================================= */

let orbitAngle =
  0;

let orbitDepth =
  0;


/* =========================================================
   EFFECT STATE
========================================================= */

const enabledEffects =
  Object.fromEntries(
    MODEL_REGISTRY.map(
      rawConfig => {

        const config =
          normalizeModelConfig(
            rawConfig
          );

        return [
          config.id,
          config.enabled
        ];
      }
    )
  );

enabledEffects.stars =
  true;


/* =========================================================
   TRACKED BODY
========================================================= */

const trackedBody = {

  valid: false,

  centerX: 0.5,
  centerY: 0.5,

  shoulderWidth: 0.25,
  torsoHeight: 0.35,

  leftShoulderX: 0.4,
  leftShoulderY: 0.35,

  rightShoulderX: 0.6,
  rightShoulderY: 0.35,

  headX: 0.5,
  headY: 0.25
};


/* =========================================================
   LANDMARK CONSTANTS
========================================================= */

const LANDMARK = {

  NOSE: 0,

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_HIP: 23,
  RIGHT_HIP: 24
};


/* =========================================================
   ORBIT CONSTANTS
========================================================= */

const POSITION_SMOOTHING =
  10;

const SCALE_SMOOTHING =
  8;

const ROTATION_SMOOTHING =
  8;

const ORBIT_SPEED =
  1.6;

const ORBIT_RADIUS_X =
  1.35;

const ORBIT_RADIUS_Y =
  0.38;

const ORBIT_CENTER_Y =
  0.12;

const ORBIT_DEPTH_SCALE =
  0.25;

const ORBIT_BACK_WIDTH_BOOST =
  0.28;

const ORBIT_BACK_LIFT =
  0.18;

const ORBIT_OCCLUSION_THRESHOLD =
  -0.72;

const ORBIT_EDGE_THRESHOLD =
  -0.25;

const FOLLOW_ORBIT_DIRECTION =
  true;


/* =========================================================
   SEGMENTATION CONSTANTS
========================================================= */

const SEGMENTATION_INTERVAL =
  1000 / 20;

const SEGMENTATION_LOW_THRESHOLD =
  0.12;

const SEGMENTATION_HIGH_THRESHOLD =
  0.52;

const SEGMENTATION_EXPANSION =
  1.02;

const SEGMENTATION_TEMPORAL_ALPHA =
  0.70;


/* =========================================================
   TEMP CANVASES
========================================================= */

const segmentationCanvas =
  document.createElement(
    "canvas"
  );

const segmentationCtx =
  segmentationCanvas.getContext(
    "2d"
  );

const compositeCanvas =
  document.createElement(
    "canvas"
  );

const compositeCtx =
  compositeCanvas.getContext(
    "2d"
  );

const maskCanvas =
  document.createElement(
    "canvas"
  );

const maskCtx =
  maskCanvas.getContext(
    "2d"
  );


/* =========================================================
   CAPTURE CANVAS
========================================================= */

const captureCanvas =
  document.createElement(
    "canvas"
  );

const captureCtx =
  captureCanvas.getContext(
    "2d"
  );


/* =========================================================
   STARS
========================================================= */

const starParticles =
  [];

const STAR_COUNT =
  14;


/* =========================================================
   BASIC HELPERS
========================================================= */

function clamp(
  value,
  min,
  max
) {

  return Math.min(
    max,
    Math.max(
      min,
      value
    )
  );
}


function lerp(
  start,
  end,
  amount
) {

  return (
    start +
    (
      end -
      start
    ) *
    amount
  );
}


function damp(
  current,
  target,
  lambda,
  delta
) {

  return THREE.MathUtils.damp(
    current,
    target,
    lambda,
    delta
  );
}

function dampAngle(
  current,
  target,
  lambda,
  delta
) {

  let difference =
    target -
    current;


  while (
    difference >
    Math.PI
  ) {

    difference -=
      Math.PI * 2;
  }


  while (
    difference <
    -Math.PI
  ) {

    difference +=
      Math.PI * 2;
  }


  return damp(
    current,
    current + difference,
    lambda,
    delta
  );
}


function setError(
  error
) {

  console.error(
    error
  );

  errorStatus.textContent =
    error?.message ||
    String(
      error
    );
}


function clearError() {

  errorStatus.textContent =
    "None";
}


/* =========================================================
   EFFECT HELPERS
========================================================= */

function effectIsEnabled(
  modelId
) {

  return Boolean(
    enabledEffects[
    modelId
    ]
  );
}


function createModelToggles() {

  modelToggleElements.clear();

  modelToggleContainer.innerHTML =
    "";


  const configs =
    MODEL_REGISTRY
      .map(
        rawConfig =>
          normalizeModelConfig(
            rawConfig
          )
      )
      .filter(
        config =>
          config.toggleable
      )
      .sort(
        (
          a,
          b
        ) =>
          a.ui.order -
          b.ui.order
      );


  for (
    const config
    of configs
  ) {

    const label =
      document.createElement(
        "label"
      );

    const toggleElement =
      document.createElement(
        "input"
      );


    toggleElement.type =
      "checkbox";

    toggleElement.id =
      `toggle-${config.id}`;

    toggleElement.checked =
      effectIsEnabled(
        config.id
      );


    label.appendChild(
      toggleElement
    );

    label.append(
      ` ${config.ui.label}`
    );


    modelToggleContainer.appendChild(
      label
    );

    modelToggleElements.set(
      config.id,
      toggleElement
    );
  }
}


/* =========================================================
   BODY / SCREEN COORDINATES
========================================================= */

function getVideoTransform() {

  const videoRect =
    video.getBoundingClientRect();


  const sourceWidth =
    video.videoWidth;

  const sourceHeight =
    video.videoHeight;


  if (
    !sourceWidth ||
    !sourceHeight ||
    !videoRect.width ||
    !videoRect.height
  ) {

    return null;
  }


  const sourceAspect =
    sourceWidth /
    sourceHeight;

  const targetAspect =
    videoRect.width /
    videoRect.height;


  let renderedWidth;
  let renderedHeight;
  let offsetX;
  let offsetY;


  if (
    sourceAspect >
    targetAspect
  ) {

    renderedHeight =
      videoRect.height;

    renderedWidth =
      renderedHeight *
      sourceAspect;

    offsetX =
      (
        videoRect.width -
        renderedWidth
      ) /
      2;

    offsetY =
      0;

  } else {

    renderedWidth =
      videoRect.width;

    renderedHeight =
      renderedWidth /
      sourceAspect;

    offsetX =
      0;

    offsetY =
      (
        videoRect.height -
        renderedHeight
      ) /
      2;
  }


  return {

    videoRect,

    renderedWidth,
    renderedHeight,

    offsetX,
    offsetY
  };
}


function mapLandmarkToScreen(
  landmark
) {

  const transform =
    getVideoTransform();


  if (
    !transform
  ) {

    return null;
  }


  let normalizedX =
    landmark.x;


  if (
    facingMode ===
    "user"
  ) {

    normalizedX =
      1 -
      normalizedX;
  }


  const x =
    (
      transform.offsetX +
      normalizedX *
      transform.renderedWidth
    ) /
    transform.videoRect.width;


  const y =
    (
      transform.offsetY +
      landmark.y *
      transform.renderedHeight
    ) /
    transform.videoRect.height;


  return {
    x,
    y
  };
}


function landmarkReliable(
  landmark
) {

  if (
    !landmark
  ) {

    return false;
  }


  const visibility =
    landmark.visibility ??
    1;


  const presence =
    landmark.presence ??
    1;


  return (
    visibility >=
    0.45 &&
    presence >=
    0.45
  );
}


/* =========================================================
   UPDATE TRACKED BODY
========================================================= */

function updateTrackedBody(
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

  const nose =
    landmarks[
    LANDMARK.NOSE
    ];


  if (
    !landmarkReliable(
      leftShoulder
    ) ||
    !landmarkReliable(
      rightShoulder
    ) ||
    !landmarkReliable(
      leftHip
    ) ||
    !landmarkReliable(
      rightHip
    )
  ) {

    trackedBody.valid =
      false;

    return;
  }


  const leftShoulderPoint =
    mapLandmarkToScreen(
      leftShoulder
    );

  const rightShoulderPoint =
    mapLandmarkToScreen(
      rightShoulder
    );

  const leftHipPoint =
    mapLandmarkToScreen(
      leftHip
    );

  const rightHipPoint =
    mapLandmarkToScreen(
      rightHip
    );


  if (
    !leftShoulderPoint ||
    !rightShoulderPoint ||
    !leftHipPoint ||
    !rightHipPoint
  ) {

    trackedBody.valid =
      false;

    return;
  }


  const shoulderCenterX =
    (
      leftShoulderPoint.x +
      rightShoulderPoint.x
    ) /
    2;

  const shoulderCenterY =
    (
      leftShoulderPoint.y +
      rightShoulderPoint.y
    ) /
    2;


  const hipCenterX =
    (
      leftHipPoint.x +
      rightHipPoint.x
    ) /
    2;

  const hipCenterY =
    (
      leftHipPoint.y +
      rightHipPoint.y
    ) /
    2;


  trackedBody.centerX =
    (
      shoulderCenterX +
      hipCenterX
    ) /
    2;

  trackedBody.centerY =
    (
      shoulderCenterY +
      hipCenterY
    ) /
    2;


  trackedBody.shoulderWidth =
    Math.abs(
      rightShoulderPoint.x -
      leftShoulderPoint.x
    );


  trackedBody.torsoHeight =
    Math.abs(
      hipCenterY -
      shoulderCenterY
    );


  if (
    leftShoulderPoint.x <
    rightShoulderPoint.x
  ) {

    trackedBody.leftShoulderX =
      leftShoulderPoint.x;

    trackedBody.leftShoulderY =
      leftShoulderPoint.y;

    trackedBody.rightShoulderX =
      rightShoulderPoint.x;

    trackedBody.rightShoulderY =
      rightShoulderPoint.y;

  } else {

    trackedBody.leftShoulderX =
      rightShoulderPoint.x;

    trackedBody.leftShoulderY =
      rightShoulderPoint.y;

    trackedBody.rightShoulderX =
      leftShoulderPoint.x;

    trackedBody.rightShoulderY =
      leftShoulderPoint.y;
  }


  if (
    landmarkReliable(
      nose
    )
  ) {

    const headPoint =
      mapLandmarkToScreen(
        nose
      );


    if (
      headPoint
    ) {

      trackedBody.headX =
        headPoint.x;

      trackedBody.headY =
        headPoint.y;
    }
  }


  trackedBody.valid =
    true;
}


/* =========================================================
   CANVAS SIZE
========================================================= */

function resizeOverlay() {

  const rect =
    video.getBoundingClientRect();


  if (
    rect.width <= 0 ||
    rect.height <= 0
  ) {

    return;
  }


  const width =
    Math.round(
      rect.width
    );

  const height =
    Math.round(
      rect.height
    );


  if (
    overlay.width !==
    width ||
    overlay.height !==
    height
  ) {

    overlay.width =
      width;

    overlay.height =
      height;
  }


  if (
    effectOverlay.width !==
    width ||
    effectOverlay.height !==
    height
  ) {

    effectOverlay.width =
      width;

    effectOverlay.height =
      height;

    initializeStars();
  }
}


/* =========================================================
   CLEAR OVERLAYS
========================================================= */

function clearOverlay() {

  ctx.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );
}


function clearEffectOverlay() {

  effectCtx.clearRect(
    0,
    0,
    effectOverlay.width,
    effectOverlay.height
  );
}


/* =========================================================
   CAMERA SUPPORT
========================================================= */

function stopMediaTracks() {

  if (
    stream
  ) {

    for (
      const track
      of stream.getTracks()
    ) {

      track.stop();
    }
  }


  stream =
    null;
}


function updateVideoMirror() {

  if (
    facingMode ===
    "user"
  ) {

    video.classList.add(
      "mirrored"
    );

  } else {

    video.classList.remove(
      "mirrored"
    );
  }
}


/* =========================================================
   CAMERA START
========================================================= */

async function startCamera() {

  try {

    clearError();


    if (
      cameraRunning
    ) {

      return;
    }


    startButton.disabled =
      true;


    cameraStatus.textContent =
      "Starting...";


    stopMediaTracks();


    stream =
      await navigator.mediaDevices.getUserMedia({

        audio: false,

        video: {

          facingMode: {
            ideal:
              facingMode
          },

          width: {
            ideal:
              1280
          },

          height: {
            ideal:
              720
          }
        }
      });


    video.srcObject =
      stream;


    await video.play();


    updateVideoMirror();


    cameraRunning =
      true;


    video.style.display =
      "block";

    threeLayer.style.display =
      "block";

    frontThreeLayer.style.display =
      "block";

    effectOverlay.style.display =
      "block";

    overlay.style.display =
      "block";


    stopButton.disabled =
      false;

    switchButton.disabled =
      false;

    captureButton.disabled =
      !systemReady();


    cameraStatus.textContent =
      facingMode ===
        "user"
        ? "Running — Front"
        : "Running — Rear";


    resizeOverlay();

    resizeThree();


    lastVideoTime =
      -1;

    lastFrameTimestamp =
      performance.now();


    cancelAnimationFrame(
      animationFrameId
    );


    animationFrameId =
      requestAnimationFrame(
        predictPose
      );

  } catch (
  error
  ) {

    cameraRunning =
      false;

    cameraStatus.textContent =
      "Camera Error";

    startButton.disabled =
      false;

    stopButton.disabled =
      true;

    switchButton.disabled =
      true;

    captureButton.disabled =
      true;


    setError(
      error
    );
  }
}


/* =========================================================
   CAMERA STOP
========================================================= */

function stopStream() {

  cameraRunning =
    false;


  if (
    animationFrameId
  ) {

    cancelAnimationFrame(
      animationFrameId
    );

    animationFrameId =
      null;
  }


  stopMediaTracks();


  video.srcObject =
    null;


  video.style.display =
    "none";

  threeLayer.style.display =
    "none";

  frontThreeLayer.style.display =
    "none";

  effectOverlay.style.display =
    "none";

  overlay.style.display =
    "none";


  clearOverlay();

  clearEffectOverlay();

  hideAllModels();


  latestLandmarks =
    null;

  latestSegmentationMask =
    null;

  trackedBody.valid =
    false;


  cameraStatus.textContent =
    "Stopped";

  poseStatus.textContent =
    "Waiting";

  segmentationStatus.textContent =
    "Waiting";

  anchorStatus.textContent =
    "Hidden";


  startButton.disabled =
    false;

  stopButton.disabled =
    true;

  switchButton.disabled =
    false;

  captureButton.disabled =
    true;
}


/* =========================================================
   SWITCH CAMERA
========================================================= */

async function switchCamera() {

  facingMode =
    facingMode ===
      "user"
      ? "environment"
      : "user";


  if (
    !cameraRunning
  ) {

    updateVideoMirror();

    return;
  }


  stopStream();

  await startCamera();
}


/* =========================================================
   THREE RESIZE
========================================================= */

function resizeThree() {

  if (
    !renderer ||
    !frontRenderer ||
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


  frontRenderer.setSize(
    rect.width,
    rect.height,
    false
  );


  threeCamera.left =
    0;

  threeCamera.right =
    1;

  threeCamera.top =
    1;

  threeCamera.bottom =
    0;


  threeCamera.updateProjectionMatrix();
}


/* =========================================================
   THREE INITIALIZATION
========================================================= */

function initializeThree() {

  scene =
    new THREE.Scene();


  threeCamera =
    new THREE.OrthographicCamera(
      0,
      1,
      1,
      0,
      -10,
      10
    );


  threeCamera.position.z =
    5;


  renderer =
    new THREE.WebGLRenderer({

      alpha: true,

      antialias: true,

      preserveDrawingBuffer: true
    });


  renderer.setClearColor(
    0x000000,
    0
  );


  renderer.setPixelRatio(
    Math.min(
      window.devicePixelRatio ||
      1,
      2
    )
  );


  renderer.outputColorSpace =
    THREE.SRGBColorSpace;


  threeLayer.appendChild(
    renderer.domElement
  );


  frontRenderer =
    new THREE.WebGLRenderer({

      alpha: true,

      antialias: true,

      preserveDrawingBuffer: true
    });


  frontRenderer.setClearColor(
    0x000000,
    0
  );


  frontRenderer.setPixelRatio(
    Math.min(
      window.devicePixelRatio ||
      1,
      2
    )
  );


  frontRenderer.outputColorSpace =
    THREE.SRGBColorSpace;


  frontThreeLayer.appendChild(
    frontRenderer.domElement
  );


  const ambient =
    new THREE.AmbientLight(
      0xffffff,
      2.2
    );


  scene.add(
    ambient
  );


  const directional =
    new THREE.DirectionalLight(
      0xffffff,
      2.5
    );


  directional.position.set(
    1,
    2,
    4
  );


  scene.add(
    directional
  );


  resizeThree();


  renderer.render(
    scene,
    threeCamera
  );


  frontRenderer.render(
    scene,
    threeCamera
  );


  threeStatus.textContent =
    `Ready r${THREE.REVISION}`;
}

/* =========================================================
   MODEL CONFIG VALIDATION
========================================================= */

function validateModelConfig(
  config
) {

  if (
    !config ||
    !config.id
  ) {

    throw new Error(
      "Invalid model config"
    );
  }


  if (
    !config.path ||
    !config.path
      .toLowerCase()
      .endsWith(".glb")
  ) {

    throw new Error(
      `${config.id}: invalid GLB path`
    );
  }


  if (
    ![
      "ORBIT",
      "BESIDE",
      "SHOULDER",
      "TORSO_ATTACH",
      "HEAD_ATTACH"
    ].includes(
      config.behavior
    )
  ) {

    throw new Error(
      `${config.id}: unsupported behavior ${config.behavior}`
    );
  }


  if (
    !Number.isFinite(
      config.scaleMultiplier
    )
  ) {

    throw new Error(
      `${config.id}: invalid scale`
    );
  }
}


/* =========================================================
   LOAD MODEL
========================================================= */

function loadModelInstance(
  config
) {

  return new Promise(
    (
      resolve,
      reject
    ) => {

      try {

        validateModelConfig(
          config
        );

      } catch (error) {

        reject(
          error
        );

        return;
      }


      const loader =
        new GLTFLoader();


      loader.load(

        config.path,

        gltf => {

          try {

            resolve(
              createModelInstance(
                gltf,
                config
              )
            );

          } catch (error) {

            reject(
              error
            );
          }
        },

        undefined,

        () => {

          reject(
            new Error(
              `Unable to load ${config.path}`
            )
          );
        }
      );
    }
  );
}


/* =========================================================
   CREATE MODEL INSTANCE
========================================================= */

function createModelInstance(
  gltf,
  config
) {

  if (
    !gltf.scene
  ) {

    throw new Error(
      `${config.name}: no scene`
    );
  }


  let meshCount =
    0;


  const materials =
    new Set();


  gltf.scene.traverse(
    object => {

      if (
        !object.isMesh
      ) {

        return;
      }


      meshCount++;


      object.frustumCulled =
        false;


      if (
        !object.material
      ) {

        return;
      }


      const list =
        Array.isArray(
          object.material
        )
          ? object.material
          : [
            object.material
          ];


      for (
        const material
        of list
      ) {

        materials.add(
          material
        );
      }
    }
  );


  if (
    meshCount === 0
  ) {

    throw new Error(
      `${config.name}: no mesh`
    );
  }


  const box =
    new THREE.Box3()
      .setFromObject(
        gltf.scene
      );


  if (
    box.isEmpty()
  ) {

    throw new Error(
      `${config.name}: empty bounding box`
    );
  }


  const size =
    new THREE.Vector3();


  const center =
    new THREE.Vector3();


  box.getSize(
    size
  );


  box.getCenter(
    center
  );


  const maxDimension =
    Math.max(
      size.x,
      size.y,
      size.z
    );


  if (
    !Number.isFinite(
      maxDimension
    ) ||
    maxDimension <= 0
  ) {

    throw new Error(
      `${config.name}: invalid model size`
    );
  }


  const root =
    gltf.scene;


  root.position.set(
    -center.x,
    -center.y,
    -center.z
  );


  root.scale.setScalar(
    1 /
    maxDimension
  );


  root.rotation.set(
    config.rotation?.x || 0,
    config.rotation?.y || 0,
    config.rotation?.z || 0
  );


  const anchor =
    new THREE.Group();


  anchor.visible =
    false;


  anchor.position.set(
    0.5,
    0.5,
    0
  );


  anchor.scale.setScalar(
    0.1
  );


  anchor.add(
    root
  );


  scene.add(
    anchor
  );


  const animations =
    gltf.animations || [];


  const clipNames =
    animations.map(
      (
        clip,
        index
      ) =>
        clip.name ||
        `Clip ${index}`
    );


  console.log(
    `[${config.name}] animations:`,
    clipNames
  );


  let mixer =
    null;


  let action =
    null;


  let selectedClipName =
    "None";


  if (
    animations.length > 0
  ) {

    const requestedIndex =
      Number.isInteger(
        config.animationIndex
      )
        ? config.animationIndex
        : 0;


    if (
      requestedIndex < 0 ||
      requestedIndex >=
      animations.length
    ) {

      throw new Error(
        `${config.name}: invalid animationIndex ${requestedIndex}`
      );
    }


    const clip =
      animations[
      requestedIndex
      ];


    selectedClipName =
      clip.name ||
      `Clip ${requestedIndex}`;


    mixer =
      new THREE.AnimationMixer(
        root
      );


    action =
      mixer.clipAction(
        clip
      );


    action.reset();


    action.setLoop(
      THREE.LoopRepeat,
      Infinity
    );


    action.play();
  }


  return {

    config,

    root,

    anchor,

    mixer,

    action,

    stats: {

      meshes:
        meshCount,

      materials:
        materials.size,

      size,

      animations:
        animations.length,

      clipNames,

      selectedClipName
    }
  };
}


/* =========================================================
   LOAD ALL MODELS
========================================================= */

async function loadAllModels() {

  allModelsReady =
    false;


  try {

    const instances =
      await Promise.all(
        MODEL_REGISTRY.map(
          rawConfig => {

            const config =
              normalizeModelConfig(
                rawConfig
              );


            return loadModelInstance(
              config
            );
          }
        )
      );


    modelInstances.clear();


    for (
      const instance
      of instances
    ) {

      modelInstances.set(
        instance.config.id,
        instance
      );
    }


    allModelsReady =
      true;


    updateCombinedModelDebug();

    updateControls();

    clearError();

  } catch (
  error
  ) {

    allModelsReady =
      false;


    modelStatus.textContent =
      "FAILED";


    setError(
      error
    );
  }
}


/* =========================================================
   MODEL DEBUG
========================================================= */

function updateCombinedModelDebug() {

  const instances =
    Array.from(
      modelInstances.values()
    );


  const totalMeshes =
    instances.reduce(
      (
        total,
        item
      ) =>
        total +
        item.stats.meshes,
      0
    );


  const totalMaterials =
    instances.reduce(
      (
        total,
        item
      ) =>
        total +
        item.stats.materials,
      0
    );


  const totalAnimations =
    instances.reduce(
      (
        total,
        item
      ) =>
        total +
        item.stats.animations,
      0
    );


  const names =
    instances.map(
      instance =>
        `${instance.config.name}: ${instance.stats.selectedClipName}`
    );


  modelStatus.textContent =
    [
      `${instances.length}/${MODEL_REGISTRY.length} models`,
      `${totalMeshes} meshes`,
      `${totalMaterials} materials`,
      `${totalAnimations} animations`,
      names.join(" | ")
    ].join(" — ");
}


/* =========================================================
   MODEL VISIBILITY
========================================================= */

function hideAllModels() {

  for (
    const instance
    of modelInstances.values()
  ) {

    instance.anchor.visible =
      false;
  }
}


function updateEffectVisibility() {

  for (
    const instance
    of modelInstances.values()
  ) {

    if (
      !effectIsEnabled(
        instance.config.id
      )
    ) {

      instance.anchor.visible =
        false;
    }
  }
}


/* =========================================================
   MODEL ANIMATIONS
========================================================= */

function updateModelAnimations(
  delta
) {

  for (
    const instance
    of modelInstances.values()
  ) {

    if (
      instance.mixer
    ) {

      instance.mixer.update(
        delta
      );
    }
  }
}


/* =========================================================
   CAPTURE EFFECT NAME
========================================================= */

function getCaptureEffectName() {

  const names =
    [];


  for (
    const rawConfig
    of MODEL_REGISTRY
  ) {

    const config =
      normalizeModelConfig(
        rawConfig
      );


    if (
      effectIsEnabled(
        config.id
      )
    ) {

      names.push(
        config.id
      );
    }
  }


  if (
    enabledEffects.stars
  ) {

    names.push(
      "stars"
    );
  }


  if (
    names.length ===
    0
  ) {

    return "no-effects";
  }


  return names.join(
    "-"
  );
}


/* =========================================================
   MODEL TOGGLE HANDLER
========================================================= */

function handleModelToggle(
  modelId,
  checked
) {

  enabledEffects[
    modelId
  ] =
    checked;


  const instance =
    modelInstances.get(
      modelId
    );


  if (
    instance &&
    !checked
  ) {

    instance.anchor.visible =
      false;
  }


  updateEffectDebug();
}


/* =========================================================
   EFFECT DEBUG
========================================================= */

function updateEffectDebug() {

  const active =
    [];


  for (
    const rawConfig
    of MODEL_REGISTRY
  ) {

    const config =
      normalizeModelConfig(
        rawConfig
      );


    if (
      effectIsEnabled(
        config.id
      )
    ) {

      active.push(
        config.ui.label
      );
    }
  }


  if (
    enabledEffects.stars
  ) {

    active.push(
      "Stars"
    );
  }


  effectStatus.textContent =
    active.length > 0
      ? active.join(
        " + "
      )
      : "None";
}


/* =========================================================
   UPDATE CONTROLS
========================================================= */

function updateControls() {

  startButton.disabled =
    cameraRunning;


  stopButton.disabled =
    !cameraRunning;


  switchButton.disabled =
    false;


  captureButton.disabled =
    !(
      cameraRunning &&
      systemReady()
    );


  for (
    const rawConfig
    of MODEL_REGISTRY
  ) {

    const config =
      normalizeModelConfig(
        rawConfig
      );


    const toggle =
      document.querySelector(
        `#toggle-${config.id}`
      );


    if (
      toggle
    ) {

      toggle.checked =
        effectIsEnabled(
          config.id
        );
    }
  }


  if (
    toggleStars
  ) {

    toggleStars.checked =
      enabledEffects.stars;
  }


  updateEffectDebug();
}


/* =========================================================
   SYSTEM READY
========================================================= */

function systemReady() {

  return Boolean(
    poseLandmarker &&
    imageSegmenter &&
    renderer &&
    frontRenderer &&
    allModelsReady
  );
}


/* =========================================================
   MEDIAPIPE INITIALIZATION
========================================================= */

async function initializeMediaPipe() {

  try {

    poseStatus.textContent =
      "Loading...";


    segmentationStatus.textContent =
      "Loading...";


    const vision =
      await FilesetResolver.forVisionTasks(

        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
      );


    poseLandmarker =
      await PoseLandmarker.createFromOptions(

        vision,

        {

          baseOptions: {

            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",

            delegate:
              "GPU"
          },

          runningMode:
            "VIDEO",

          numPoses:
            1,

          minPoseDetectionConfidence:
            0.5,

          minPosePresenceConfidence:
            0.5,

          minTrackingConfidence:
            0.5
        }
      );


    poseStatus.textContent =
      "Ready";


    imageSegmenter =
      await ImageSegmenter.createFromOptions(

        vision,

        {

          baseOptions: {

            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",

            delegate:
              "GPU"
          },

          runningMode:
            "VIDEO",

          outputCategoryMask:
            false,

          outputConfidenceMasks:
            true
        }
      );


    segmentationStatus.textContent =
      "Ready";


    updateControls();

    clearError();

  } catch (
  error
  ) {

    poseStatus.textContent =
      "FAILED";

    segmentationStatus.textContent =
      "FAILED";


    setError(
      error
    );
  }
}


/* =========================================================
   POSE DEBUG
========================================================= */

function drawPoseDebug(
  landmarks
) {

  if (
    !landmarks ||
    landmarks.length ===
    0
  ) {

    return;
  }


  const connections = [

    [
      LANDMARK.LEFT_SHOULDER,
      LANDMARK.RIGHT_SHOULDER
    ],

    [
      LANDMARK.LEFT_SHOULDER,
      LANDMARK.LEFT_HIP
    ],

    [
      LANDMARK.RIGHT_SHOULDER,
      LANDMARK.RIGHT_HIP
    ],

    [
      LANDMARK.LEFT_HIP,
      LANDMARK.RIGHT_HIP
    ]
  ];


  ctx.save();


  ctx.strokeStyle =
    "#00ff88";

  ctx.fillStyle =
    "#00ff88";

  ctx.lineWidth =
    3;


  for (
    const [
      startIndex,
      endIndex
    ]
    of connections
  ) {

    const start =
      mapLandmarkToScreen(
        landmarks[
        startIndex
        ]
      );

    const end =
      mapLandmarkToScreen(
        landmarks[
        endIndex
        ]
      );


    if (
      !start ||
      !end
    ) {

      continue;
    }


    ctx.beginPath();


    ctx.moveTo(
      start.x *
      overlay.width,

      start.y *
      overlay.height
    );


    ctx.lineTo(
      end.x *
      overlay.width,

      end.y *
      overlay.height
    );


    ctx.stroke();
  }


  const pointIndexes = [

    LANDMARK.NOSE,

    LANDMARK.LEFT_SHOULDER,
    LANDMARK.RIGHT_SHOULDER,

    LANDMARK.LEFT_HIP,
    LANDMARK.RIGHT_HIP
  ];


  for (
    const index
    of pointIndexes
  ) {

    const point =
      mapLandmarkToScreen(
        landmarks[
        index
        ]
      );


    if (
      !point
    ) {

      continue;
    }


    ctx.beginPath();


    ctx.arc(
      point.x *
      overlay.width,

      point.y *
      overlay.height,

      5,

      0,

      Math.PI *
      2
    );


    ctx.fill();
  }


  ctx.restore();
}


/* =========================================================
   SEGMENTATION MASK
========================================================= */

function buildHumanOcclusionLayer() {

  if (
    !latestSegmentationMask ||
    !cameraRunning ||
    video.readyState <
    2
  ) {

    return false;
  }


  const width =
    overlay.width;

  const height =
    overlay.height;


  if (
    !width ||
    !height
  ) {

    return false;
  }


  compositeCanvas.width =
    width;

  compositeCanvas.height =
    height;


  maskCanvas.width =
    width;

  maskCanvas.height =
    height;


  segmentationCanvas.width =
    latestSegmentationMask.width;

  segmentationCanvas.height =
    latestSegmentationMask.height;


  const maskData =
    latestSegmentationMask
      .getAsFloat32Array();


  const imageData =
    segmentationCtx.createImageData(
      latestSegmentationMask.width,
      latestSegmentationMask.height
    );


  for (
    let i = 0;
    i <
    maskData.length;
    i++
  ) {

    const confidence =
      maskData[
      i
      ];


    const alpha =
      confidence <=
        SEGMENTATION_LOW_THRESHOLD
        ? 0
        : confidence >=
          SEGMENTATION_HIGH_THRESHOLD
          ? 255
          : Math.round(
            (
              (
                confidence -
                SEGMENTATION_LOW_THRESHOLD
              ) /
              (
                SEGMENTATION_HIGH_THRESHOLD -
                SEGMENTATION_LOW_THRESHOLD
              )
            ) *
            255
          );


    const offset =
      i *
      4;


    imageData.data[
      offset
    ] =
      255;

    imageData.data[
      offset +
      1
    ] =
      255;

    imageData.data[
      offset +
      2
    ] =
      255;

    imageData.data[
      offset +
      3
    ] =
      alpha;
  }


  segmentationCtx.putImageData(
    imageData,
    0,
    0
  );


  maskCtx.clearRect(
    0,
    0,
    width,
    height
  );


  const transform =
    getVideoTransform();


  if (
    !transform
  ) {

    return false;
  }


  maskCtx.save();


  if (
    facingMode ===
    "user"
  ) {

    maskCtx.translate(
      width,
      0
    );

    maskCtx.scale(
      -1,
      1
    );
  }


  maskCtx.drawImage(
    segmentationCanvas,

    0,
    0,

    segmentationCanvas.width,
    segmentationCanvas.height,

    transform.offsetX,
    transform.offsetY,

    transform.renderedWidth,
    transform.renderedHeight
  );


  maskCtx.restore();


  compositeCtx.clearRect(
    0,
    0,
    width,
    height
  );


  compositeCtx.save();


  compositeCtx.drawImage(
    video,

    0,
    0,

    video.videoWidth,
    video.videoHeight,

    transform.offsetX,
    transform.offsetY,

    transform.renderedWidth,
    transform.renderedHeight
  );


  if (
    facingMode ===
    "user"
  ) {

    const image =
      compositeCtx.getImageData(
        0,
        0,
        width,
        height
      );


    compositeCtx.clearRect(
      0,
      0,
      width,
      height
    );


    compositeCtx.save();

    compositeCtx.translate(
      width,
      0
    );

    compositeCtx.scale(
      -1,
      1
    );

    compositeCtx.putImageData(
      image,
      0,
      0
    );

    compositeCtx.restore();
  }


  compositeCtx.globalCompositeOperation =
    "destination-in";


  compositeCtx.drawImage(
    maskCanvas,
    0,
    0
  );


  compositeCtx.globalCompositeOperation =
    "source-over";


  return true;
}

/* =========================================================
   PER-MODEL HUMAN OCCLUSION
========================================================= */

function modelNeedsHumanOcclusion(
  instance
) {

  if (
    !instance ||
    !instance.config
  ) {

    return false;
  }


  const config =
    normalizeModelConfig(
      instance.config
    );


  if (
    !effectIsEnabled(
      config.id
    )
  ) {

    return false;
  }


  if (
    config.occlusion !==
    "HUMAN"
  ) {

    return false;
  }


  switch (
  config.behavior
  ) {

    case "ORBIT":

      return (
        orbitDepth <
        ORBIT_OCCLUSION_THRESHOLD
      );


    default:

      return false;
  }
}


/* =========================================================
   SHOULD DRAW HUMAN OCCLUSION
========================================================= */

function shouldUseHumanOcclusion() {

  for (
    const instance
    of modelInstances.values()
  ) {

    if (
      modelNeedsHumanOcclusion(
        instance
      )
    ) {

      return true;
    }
  }


  return false;
}


/* =========================================================
   DRAW HUMAN OCCLUSION
========================================================= */

function drawFullHumanOcclusion() {

  if (
    !shouldUseHumanOcclusion()
  ) {

    return;
  }


  if (
    !buildHumanOcclusionLayer()
  ) {

    return;
  }


  ctx.drawImage(
    compositeCanvas,
    0,
    0
  );
}


/* =========================================================
   ORBIT BEHAVIOR
========================================================= */

function updateOrbitBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  orbitAngle +=
    ORBIT_SPEED *
    delta;


  if (
    orbitAngle >
    Math.PI * 2
  ) {

    orbitAngle -=
      Math.PI * 2;
  }


  /* ---------------------------------------------------------
     ORBIT AXES
  --------------------------------------------------------- */

  const orbitX =
    Math.cos(
      orbitAngle
    );


  orbitDepth =
    Math.sin(
      orbitAngle
    );


  const orbitY =
    Math.sin(
      orbitAngle * 2
    );


  /* ---------------------------------------------------------
     BACK AMOUNT

     FRONT:
     orbitDepth >= 0
     backAmount = 0

     BACK:
     orbitDepth < 0
     backAmount grows toward 1
  --------------------------------------------------------- */

  const backAmount =
    Math.max(
      0,
      -orbitDepth
    );


  /* ---------------------------------------------------------
     ORBIT SIZE
  --------------------------------------------------------- */

  const radiusX =
    trackedBody.shoulderWidth *
    (
      ORBIT_RADIUS_X +
      (
        backAmount *
        ORBIT_BACK_WIDTH_BOOST
      )
    );


  const radiusY =
    trackedBody.torsoHeight *
    ORBIT_RADIUS_Y;


  const centerYOffset =
    trackedBody.torsoHeight *
    ORBIT_CENTER_Y;


  const backLift =
    trackedBody.torsoHeight *
    backAmount *
    ORBIT_BACK_LIFT;


  /* ---------------------------------------------------------
     TARGET POSITION
  --------------------------------------------------------- */

  const targetX =
    trackedBody.centerX +
    orbitX *
    radiusX;


  const targetY =
    trackedBody.centerY +
    centerYOffset +
    orbitY *
    radiusY -
    backLift;


  /* ---------------------------------------------------------
     BODY-RELATIVE SCALE
  --------------------------------------------------------- */

  const bodyReference =
    trackedBody.shoulderWidth *
    0.65
    +
    trackedBody.torsoHeight *
    0.35;


  const baseScale =
    bodyReference *
    config.scaleMultiplier;


  /* ---------------------------------------------------------
     FAKE DEPTH SCALE

     FRONT  -> slightly larger
     BACK   -> slightly smaller
  --------------------------------------------------------- */

  const depthMultiplier =
    1 +
    orbitDepth *
    ORBIT_DEPTH_SCALE;


  const targetScale =
    THREE.MathUtils.clamp(

      baseScale *
      depthMultiplier,

      0.05,

      1.2
    );


  /* ---------------------------------------------------------
     SMOOTH POSITION
  --------------------------------------------------------- */

  instance.anchor.position.x =
    damp(
      instance.anchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.y =
    damp(
      instance.anchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  /* ---------------------------------------------------------
     SCREEN-SPACE FAKE DEPTH
  --------------------------------------------------------- */

  instance.anchor.position.z =
    orbitDepth *
    0.1;


  /* ---------------------------------------------------------
     SMOOTH SCALE
  --------------------------------------------------------- */

  const smoothScale =
    damp(
      instance.anchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  instance.anchor.scale.setScalar(
    smoothScale
  );


  /* ---------------------------------------------------------
     FOLLOW TRAVEL DIRECTION
  --------------------------------------------------------- */

  if (
    FOLLOW_ORBIT_DIRECTION
  ) {

    const movementX =
      -Math.sin(
        orbitAngle
      );


    const targetRotationY =
      movementX >= 0
        ? Math.PI / 2
        : -Math.PI / 2;


    instance.anchor.rotation.y =
      dampAngle(
        instance.anchor.rotation.y,
        targetRotationY,
        ROTATION_SMOOTHING,
        delta
      );
  }


  /* ---------------------------------------------------------
     SMALL BANKING MOTION
  --------------------------------------------------------- */

  const targetBank =
    THREE.MathUtils.degToRad(
      -orbitX *
      12
    );


  instance.anchor.rotation.z =
    damp(
      instance.anchor.rotation.z,
      targetBank,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.visible =
    true;


  /* ---------------------------------------------------------
     FRONT / EDGE / BACK DEBUG
  --------------------------------------------------------- */

  let depthText =
    "FRONT";


  if (
    orbitDepth <
    ORBIT_OCCLUSION_THRESHOLD
  ) {

    depthText =
      "BACK";

  } else if (
    orbitDepth <
    ORBIT_EDGE_THRESHOLD
  ) {

    depthText =
      "EDGE";
  }


  return (
    `Butterfly ORBIT ${depthText}`
  );
}


/* =========================================================
   SHOULDER BEHAVIOR
========================================================= */

function updateShoulderBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  const shoulderConfig =
    config.shoulder ||
    {};


  const requestedSide =
    shoulderConfig.side ===
      "left"
      ? "left"
      : "right";


  const offsetX =
    Number.isFinite(
      shoulderConfig.offsetX
    )
      ? shoulderConfig.offsetX
      : 0.55;


  const offsetY =
    Number.isFinite(
      shoulderConfig.offsetY
    )
      ? shoulderConfig.offsetY
      : 0.25;


  let screenLeftX;

  let screenLeftY;

  let screenRightX;

  let screenRightY;


  if (
    trackedBody.leftShoulderX <=
    trackedBody.rightShoulderX
  ) {

    screenLeftX =
      trackedBody.leftShoulderX;


    screenLeftY =
      trackedBody.leftShoulderY;


    screenRightX =
      trackedBody.rightShoulderX;


    screenRightY =
      trackedBody.rightShoulderY;

  } else {

    screenLeftX =
      trackedBody.rightShoulderX;


    screenLeftY =
      trackedBody.rightShoulderY;


    screenRightX =
      trackedBody.leftShoulderX;


    screenRightY =
      trackedBody.leftShoulderY;
  }


  let targetX;

  let targetY;


  if (
    requestedSide ===
    "left"
  ) {

    targetX =
      screenLeftX -
      trackedBody.shoulderWidth *
      offsetX;


    targetY =
      screenLeftY +
      trackedBody.torsoHeight *
      offsetY;

  } else {

    targetX =
      screenRightX +
      trackedBody.shoulderWidth *
      offsetX;


    targetY =
      screenRightY +
      trackedBody.torsoHeight *
      offsetY;
  }


  const bodyReference =
    trackedBody.shoulderWidth *
    0.65
    +
    trackedBody.torsoHeight *
    0.35;


  const shoulderScaleFactor =
    0.88;


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      config.scaleMultiplier *
      shoulderScaleFactor,

      0.04,

      0.85
    );


  instance.anchor.position.x =
    damp(
      instance.anchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.y =
    damp(
      instance.anchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.z =
    0;


  const smoothScale =
    damp(
      instance.anchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  instance.anchor.scale.setScalar(
    smoothScale
  );


  instance.anchor.rotation.y =
    dampAngle(
      instance.anchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.rotation.z =
    damp(
      instance.anchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.visible =
    true;


  return (
    `${config.name} SHOULDER ${requestedSide.toUpperCase()}`
  );
}


/* =========================================================
   TORSO ATTACH
========================================================= */

function updateTorsoAttachBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  const torsoConfig =
    config.torso ||
    {};


  const offsetX =
    Number.isFinite(
      torsoConfig.offsetX
    )
      ? torsoConfig.offsetX
      : 0;


  const offsetY =
    Number.isFinite(
      torsoConfig.offsetY
    )
      ? torsoConfig.offsetY
      : 0;


  const scaleFactor =
    Number.isFinite(
      torsoConfig.scaleFactor
    )
      ? torsoConfig.scaleFactor
      : 1;


  const targetX =
    trackedBody.centerX +
    trackedBody.shoulderWidth *
    offsetX;


  const targetY =
    trackedBody.centerY +
    trackedBody.torsoHeight *
    offsetY;


  const bodyReference =
    trackedBody.shoulderWidth *
    0.6
    +
    trackedBody.torsoHeight *
    0.4;


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      config.scaleMultiplier *
      scaleFactor,

      0.04,

      1.2
    );


  instance.anchor.position.x =
    damp(
      instance.anchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.y =
    damp(
      instance.anchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.z =
    0;


  const smoothScale =
    damp(
      instance.anchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  instance.anchor.scale.setScalar(
    smoothScale
  );


  instance.anchor.rotation.y =
    dampAngle(
      instance.anchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.rotation.z =
    damp(
      instance.anchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.visible =
    true;


  return (
    `${config.name} TORSO_ATTACH`
  );
}


/* =========================================================
   HEAD ATTACH
========================================================= */

function updateHeadAttachBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  const headConfig =
    config.head ||
    {};


  const offsetX =
    Number.isFinite(
      headConfig.offsetX
    )
      ? headConfig.offsetX
      : 0;


  const offsetY =
    Number.isFinite(
      headConfig.offsetY
    )
      ? headConfig.offsetY
      : 0.55;


  const scaleFactor =
    Number.isFinite(
      headConfig.scaleFactor
    )
      ? headConfig.scaleFactor
      : 1;


  const targetX =
    trackedBody.headX +
    trackedBody.shoulderWidth *
    offsetX;


  const targetY =
    trackedBody.headY +
    trackedBody.shoulderWidth *
    offsetY;


  const targetScale =
    THREE.MathUtils.clamp(

      trackedBody.shoulderWidth *
      config.scaleMultiplier *
      scaleFactor,

      0.04,

      0.8
    );


  instance.anchor.position.x =
    damp(
      instance.anchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.y =
    damp(
      instance.anchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.z =
    0.03;


  const smoothScale =
    damp(
      instance.anchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  instance.anchor.scale.setScalar(
    smoothScale
  );


  instance.anchor.rotation.y =
    dampAngle(
      instance.anchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.rotation.z =
    damp(
      instance.anchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.visible =
    true;


  return (
    `${config.name} HEAD_ATTACH`
  );
}


/* =========================================================
   BESIDE
========================================================= */

function updateBesideBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  const beside =
    config.beside ||
    {};


  const side =
    beside.side ===
      "left"
      ? -1
      : 1;


  const distance =
    Number.isFinite(
      beside.distance
    )
      ? beside.distance
      : 1.2;


  const offsetY =
    Number.isFinite(
      beside.offsetY
    )
      ? beside.offsetY
      : 0;


  const targetX =
    trackedBody.centerX +
    side *
    trackedBody.shoulderWidth *
    distance;


  const targetY =
    trackedBody.centerY +
    trackedBody.torsoHeight *
    offsetY;


  const bodyReference =
    trackedBody.shoulderWidth *
    0.65
    +
    trackedBody.torsoHeight *
    0.35;


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      config.scaleMultiplier,

      0.05,

      1.2
    );


  instance.anchor.position.x =
    damp(
      instance.anchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.y =
    damp(
      instance.anchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  instance.anchor.position.z =
    0;


  const smoothScale =
    damp(
      instance.anchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  instance.anchor.scale.setScalar(
    smoothScale
  );


  instance.anchor.rotation.y =
    dampAngle(
      instance.anchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.rotation.z =
    damp(
      instance.anchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  instance.anchor.visible =
    true;


  return (
    `${config.name} BESIDE`
  );
}


/* =========================================================
   ALL MODEL BEHAVIORS
========================================================= */

function updateAllModelBehaviors(
  delta
) {

  if (
    !trackedBody.valid
  ) {

    hideAllModels();


    anchorStatus.textContent =
      "Hidden";


    return;
  }


  const debug =
    [];


  let orbitActive =
    false;


  for (
    const instance
    of modelInstances.values()
  ) {

    const config =
      normalizeModelConfig(
        instance.config
      );


    if (
      !effectIsEnabled(
        config.id
      )
    ) {

      instance.anchor.visible =
        false;


      continue;
    }


    switch (
    config.behavior
    ) {

      case "ORBIT":

        orbitActive =
          true;


        debug.push(
          updateOrbitBehavior(
            instance,
            delta
          )
        );


        break;


      case "SHOULDER":

        debug.push(
          updateShoulderBehavior(
            instance,
            delta
          )
        );


        break;


      case "BESIDE":

        debug.push(
          updateBesideBehavior(
            instance,
            delta
          )
        );


        break;


      case "TORSO_ATTACH":

        debug.push(
          updateTorsoAttachBehavior(
            instance,
            delta
          )
        );


        break;


      case "HEAD_ATTACH":

        debug.push(
          updateHeadAttachBehavior(
            instance,
            delta
          )
        );


        break;


      default:

        instance.anchor.visible =
          false;

        break;
    }
  }


  if (
    !orbitActive
  ) {

    orbitDepth =
      0;
  }


  anchorStatus.textContent =
    debug.length > 0
      ? debug.join(
        " | "
      )
      : "No effects";
}

/* =========================================================
   STAR INITIALIZATION
========================================================= */

function initializeStars() {

  starParticles.length =
    0;


  for (
    let i = 0;
    i < STAR_COUNT;
    i++
  ) {

    const angle =
      (
        i /
        STAR_COUNT
      ) *
      Math.PI *
      2;


    starParticles.push({

      angle,

      speed:
        0.25 +
        Math.random() *
        0.25,

      radiusMultiplier:
        0.85 +
        Math.random() *
        0.55,

      sizeMultiplier:
        0.7 +
        Math.random() *
        0.6,

      phase:
        Math.random() *
        Math.PI *
        2
    });
  }
}


/* =========================================================
   DRAW STAR SHAPE
========================================================= */

function drawStarShape(
  context,
  x,
  y,
  outerRadius,
  innerRadius,
  rotation = 0
) {

  const points =
    5;


  context.beginPath();


  for (
    let i = 0;
    i <
    points * 2;
    i++
  ) {

    const radius =
      i % 2 === 0
        ? outerRadius
        : innerRadius;


    const angle =
      rotation +
      (
        i *
        Math.PI /
        points
      ) -
      Math.PI / 2;


    const px =
      x +
      Math.cos(
        angle
      ) *
      radius;


    const py =
      y +
      Math.sin(
        angle
      ) *
      radius;


    if (
      i === 0
    ) {

      context.moveTo(
        px,
        py
      );

    } else {

      context.lineTo(
        px,
        py
      );
    }
  }


  context.closePath();

  context.fill();
}


/* =========================================================
   DRAW STARS EFFECT
========================================================= */

function drawStarsEffect(
  timestamp
) {

  clearEffectOverlay();


  if (
    !enabledEffects.stars ||
    !trackedBody.valid
  ) {

    return;
  }


  if (
    starParticles.length ===
    0
  ) {

    initializeStars();
  }


  const width =
    effectOverlay.width;

  const height =
    effectOverlay.height;


  if (
    !width ||
    !height
  ) {

    return;
  }


  const centerX =
    trackedBody.centerX *
    width;


  const centerY =
    (
      trackedBody.centerY -
      trackedBody.torsoHeight *
      0.05
    ) *
    height;


  const bodyWidth =
    trackedBody.shoulderWidth *
    width;


  const bodyHeight =
    trackedBody.torsoHeight *
    height;


  effectCtx.save();


  effectCtx.fillStyle =
    "rgba(255,255,255,0.95)";


  for (
    const star
    of starParticles
  ) {

    const time =
      timestamp *
      0.001;


    const currentAngle =
      star.angle +
      time *
      star.speed;


    const radiusX =
      bodyWidth *
      1.3 *
      star.radiusMultiplier;


    const radiusY =
      bodyHeight *
      0.75 *
      star.radiusMultiplier;


    const pulse =
      1 +
      Math.sin(
        time *
        2.5 +
        star.phase
      ) *
      0.12;


    const x =
      centerX +
      Math.cos(
        currentAngle
      ) *
      radiusX;


    const y =
      centerY +
      Math.sin(
        currentAngle *
        1.15
      ) *
      radiusY;


    const size =
      Math.max(
        3,
        bodyWidth *
        0.055 *
        star.sizeMultiplier *
        pulse
      );


    drawStarShape(
      effectCtx,
      x,
      y,
      size,
      size *
      0.45,
      currentAngle
    );
  }


  effectCtx.restore();
}


/* =========================================================
   SEGMENTATION UPDATE
========================================================= */

function updateSegmentation(
  timestamp
) {

  if (
    !imageSegmenter ||
    !cameraRunning ||
    video.readyState <
    2
  ) {

    return;
  }


  if (
    timestamp -
    latestSegmentationTimestamp <
    SEGMENTATION_INTERVAL
  ) {

    return;
  }


  latestSegmentationTimestamp =
    timestamp;


  try {

    imageSegmenter.segmentForVideo(

      video,

      timestamp,

      result => {

        if (
          !result ||
          !result.confidenceMasks ||
          result.confidenceMasks.length ===
          0
        ) {

          segmentationStatus.textContent =
            "No Mask";


          return;
        }


        if (
          latestSegmentationMask &&
          latestSegmentationMask.close
        ) {

          latestSegmentationMask.close();
        }


        latestSegmentationMask =
          result.confidenceMasks[
          0
          ];


        segmentationStatus.textContent =
          "Human Mask";
      }
    );

  } catch (
  error
  ) {

    segmentationStatus.textContent =
      "Error";


    console.warn(
      "[Human AR] segmentation:",
      error
    );
  }
}


/* =========================================================
   DRAW OVERLAY
========================================================= */

function drawOverlay() {

  clearOverlay();


  if (
    !cameraRunning
  ) {

    return;
  }


  /*
   * IMPORTANT:
   *
   * The human cutout is drawn first.
   * Pose debug is drawn after it so that
   * the green debug skeleton stays visible.
   */

  drawFullHumanOcclusion();


  if (
    latestLandmarks
  ) {

    drawPoseDebug(
      latestLandmarks
    );
  }
}


/* =========================================================
   MODEL LAYER ROUTING

   BACK RENDERER:
   Models currently requiring HUMAN occlusion.

   FRONT RENDERER:
   Models that should stay above the human cutout.

   Example:
   Butterfly BACK -> back renderer
   Butterfly EDGE/FRONT -> front renderer
   Waveboy -> front renderer
========================================================= */

function renderModelLayers() {

  if (
    !renderer ||
    !frontRenderer ||
    !scene ||
    !threeCamera
  ) {

    return;
  }


  const originalVisibility =
    new Map();


  /*
   * Save the real visibility state produced
   * by each behavior before temporarily
   * changing visibility for the two passes.
   */

  for (
    const instance
    of modelInstances.values()
  ) {

    originalVisibility.set(
      instance.config.id,
      instance.anchor.visible
    );
  }


  /* ---------------------------------------------------------
     BACK PASS
  --------------------------------------------------------- */

  for (
    const instance
    of modelInstances.values()
  ) {

    const wasVisible =
      originalVisibility.get(
        instance.config.id
      );


    instance.anchor.visible =
      Boolean(
        wasVisible &&
        modelNeedsHumanOcclusion(
          instance
        )
      );
  }


  renderer.render(
    scene,
    threeCamera
  );


  /* ---------------------------------------------------------
     FRONT PASS
  --------------------------------------------------------- */

  for (
    const instance
    of modelInstances.values()
  ) {

    const wasVisible =
      originalVisibility.get(
        instance.config.id
      );


    instance.anchor.visible =
      Boolean(
        wasVisible &&
        !modelNeedsHumanOcclusion(
          instance
        )
      );
  }


  frontRenderer.render(
    scene,
    threeCamera
  );


  /* ---------------------------------------------------------
     RESTORE REAL MODEL VISIBILITY
  --------------------------------------------------------- */

  for (
    const instance
    of modelInstances.values()
  ) {

    instance.anchor.visible =
      Boolean(
        originalVisibility.get(
          instance.config.id
        )
      );
  }
}


/* =========================================================
   POSE RESULT
========================================================= */

function processPoseResult(
  result
) {

  if (
    !result ||
    !result.landmarks ||
    result.landmarks.length ===
    0
  ) {

    latestLandmarks =
      null;


    trackedBody.valid =
      false;


    poseStatus.textContent =
      "No Person";


    hideAllModels();


    return;
  }


  latestLandmarks =
    result.landmarks[
    0
    ];


  updateTrackedBody(
    latestLandmarks
  );


  if (
    trackedBody.valid
  ) {

    poseStatus.textContent =
      "Person Found";

  } else {

    poseStatus.textContent =
      "Pose Unstable";
  }
}


/* =========================================================
   MAIN LOOP
========================================================= */

function predictPose() {

  if (
    !cameraRunning
  ) {

    return;
  }


  animationFrameId =
    requestAnimationFrame(
      predictPose
    );


  const now =
    performance.now();


  const delta =
    Math.min(
      (
        now -
        lastFrameTimestamp
      ) /
      1000,

      0.1
    );


  lastFrameTimestamp =
    now;


  resizeOverlay();


  if (
    video.readyState <
    2
  ) {

    return;
  }


  /*
   * Run pose only when a new camera frame exists.
   */

  if (
    video.currentTime !==
    lastVideoTime
  ) {

    lastVideoTime =
      video.currentTime;


    if (
      poseLandmarker
    ) {

      try {

        const result =
          poseLandmarker.detectForVideo(
            video,
            now
          );


        processPoseResult(
          result
        );

      } catch (
      error
      ) {

        console.warn(
          "[Human AR] pose:",
          error
        );
      }
    }


    updateSegmentation(
      now
    );
  }


  /*
   * Model animations continue every render frame.
   */

  updateModelAnimations(
    delta
  );


  /*
   * Body-relative behavior update.
   */

  updateAllModelBehaviors(
    delta
  );


  updateEffectVisibility();


  /*
   * Render the GLBs into their appropriate
   * BACK or FRONT canvas.
   */

  renderModelLayers();


  /*
   * BODY_EFFECT layer.
   */

  drawStarsEffect(
    now
  );


  /*
   * Human occlusion + pose debug.
   */

  drawOverlay();
}


/* =========================================================
   CAPTURE CAMERA FRAME
========================================================= */

function drawCameraToCapture(
  width,
  height
) {

  const transform =
    getVideoTransform();


  if (
    !transform
  ) {

    return false;
  }


  captureCtx.save();


  captureCtx.clearRect(
    0,
    0,
    width,
    height
  );


  /*
   * Match the mirrored front-camera preview.
   */

  if (
    facingMode ===
    "user"
  ) {

    captureCtx.translate(
      width,
      0
    );


    captureCtx.scale(
      -1,
      1
    );
  }


  captureCtx.drawImage(

    video,

    0,
    0,

    video.videoWidth,
    video.videoHeight,

    facingMode ===
      "user"
      ? -transform.offsetX -
      transform.renderedWidth +
      width
      : transform.offsetX,

    transform.offsetY,

    transform.renderedWidth,
    transform.renderedHeight
  );


  captureCtx.restore();


  return true;
}


/* =========================================================
   CAPTURE HUMAN OCCLUSION
========================================================= */

function drawHumanOcclusionToCapture() {

  if (
    !shouldUseHumanOcclusion()
  ) {

    return;
  }


  if (
    !buildHumanOcclusionLayer()
  ) {

    return;
  }


  captureCtx.drawImage(
    compositeCanvas,
    0,
    0,

    captureCanvas.width,
    captureCanvas.height
  );
}


/* =========================================================
   CAPTURE PHOTO
========================================================= */

function capturePhoto() {

  try {

    clearError();


    if (
      !cameraRunning ||
      !systemReady() ||
      video.readyState <
      2
    ) {

      captureStatus.textContent =
        "Not Ready";


      return;
    }


    const width =
      overlay.width;


    const height =
      overlay.height;


    if (
      !width ||
      !height
    ) {

      captureStatus.textContent =
        "Invalid Size";


      return;
    }


    captureCanvas.width =
      width;


    captureCanvas.height =
      height;


    /*
     * Refresh both model render passes immediately
     * before capture.
     */

    renderModelLayers();


    /* -------------------------------------------------------
       1. CAMERA
    ------------------------------------------------------- */

    if (
      !drawCameraToCapture(
        width,
        height
      )
    ) {

      captureStatus.textContent =
        "Camera Failed";


      return;
    }


    /* -------------------------------------------------------
       2. BACK GLB LAYER

       Butterfly is here only while it is actually BACK.
    ------------------------------------------------------- */

    captureCtx.drawImage(
      renderer.domElement,

      0,
      0,

      width,
      height
    );


    /* -------------------------------------------------------
       3. HUMAN CUTOUT

       This covers only the BACK model pass.
    ------------------------------------------------------- */

    drawHumanOcclusionToCapture();


    /* -------------------------------------------------------
       4. STARS

       BODY_EFFECT currently remains above the human cutout
       in the exported photo.
    ------------------------------------------------------- */

    if (
      enabledEffects.stars
    ) {

      captureCtx.drawImage(
        effectOverlay,

        0,
        0,

        width,
        height
      );
    }


    /* -------------------------------------------------------
       5. FRONT GLB LAYER

       Waveboy and Butterfly FRONT/EDGE are here.
    ------------------------------------------------------- */

    captureCtx.drawImage(
      frontRenderer.domElement,

      0,
      0,

      width,
      height
    );


    /*
     * Pose debug is intentionally NOT captured.
     */


    captureStatus.textContent =
      "Captured";


    const effectName =
      getCaptureEffectName();


    const timestamp =
      new Date()
        .toISOString()
        .replace(
          /[:.]/g,
          "-"
        );


    const filename =
      `human-ar-${effectName}-${timestamp}.jpg`;


    captureCanvas.toBlob(

      blob => {

        if (
          !blob
        ) {

          captureStatus.textContent =
            "Capture Failed";


          return;
        }


        const url =
          URL.createObjectURL(
            blob
          );


        const link =
          document.createElement(
            "a"
          );


        link.href =
          url;


        link.download =
          filename;


        document.body.appendChild(
          link
        );


        link.click();


        link.remove();


        setTimeout(
          () => {

            URL.revokeObjectURL(
              url
            );
          },

          1000
        );
      },

      "image/jpeg",

      0.95
    );

  } catch (
  error
  ) {

    captureStatus.textContent =
      "Capture Error";


    setError(
      error
    );
  }
}

/* =========================================================
   MODEL TOGGLE EVENTS
========================================================= */

function bindModelToggleEvents() {

  for (
    const [
      modelId,
      toggleElement
    ]
    of modelToggleElements.entries()
  ) {

    toggleElement.addEventListener(
      "change",
      () => {

        handleModelToggle(
          modelId,
          toggleElement.checked
        );
      }
    );
  }
}


/* =========================================================
   STARS TOGGLE
========================================================= */

function handleStarsToggle() {

  enabledEffects.stars =
    Boolean(
      toggleStars.checked
    );


  if (
    !enabledEffects.stars
  ) {

    clearEffectOverlay();
  }


  updateEffectDebug();
}


/* =========================================================
   BUTTON EVENTS
========================================================= */

startButton.addEventListener(
  "click",
  async () => {

    await startCamera();
  }
);


stopButton.addEventListener(
  "click",
  () => {

    stopStream();
  }
);


switchButton.addEventListener(
  "click",
  async () => {

    await switchCamera();
  }
);


captureButton.addEventListener(
  "click",
  () => {

    capturePhoto();
  }
);


/* =========================================================
   STAR EVENT
========================================================= */

if (
  toggleStars
) {

  toggleStars.addEventListener(
    "change",
    handleStarsToggle
  );
}


/* =========================================================
   VIDEO EVENTS
========================================================= */

video.addEventListener(
  "loadedmetadata",
  () => {

    resizeOverlay();

    resizeThree();
  }
);


/* =========================================================
   WINDOW RESIZE
========================================================= */

window.addEventListener(
  "resize",
  () => {

    resizeOverlay();

    resizeThree();
  }
);


/* =========================================================
   ORIENTATION CHANGE
========================================================= */

window.addEventListener(
  "orientationchange",
  () => {

    /*
     * Mobile Safari can report the old viewport size
     * immediately during orientationchange.
     */

    setTimeout(
      () => {

        resizeOverlay();

        resizeThree();
      },

      250
    );
  }
);


/* =========================================================
   PAGE VISIBILITY
========================================================= */

document.addEventListener(
  "visibilitychange",
  () => {

    if (
      document.hidden
    ) {

      /*
       * Do not destroy the camera stream here.
       * Just reset timing so returning to Safari
       * does not create a huge animation delta.
       */

      lastFrameTimestamp =
        performance.now();
    }
  }
);


/* =========================================================
   BEFORE UNLOAD
========================================================= */

window.addEventListener(
  "beforeunload",
  () => {

    stopMediaTracks();


    if (
      latestSegmentationMask &&
      latestSegmentationMask.close
    ) {

      latestSegmentationMask.close();
    }
  }
);


/* =========================================================
   INITIAL UI STATE
========================================================= */

function initializeUI() {

  cameraStatus.textContent =
    "Stopped";


  poseStatus.textContent =
    "Loading...";


  segmentationStatus.textContent =
    "Loading...";


  threeStatus.textContent =
    "Loading...";


  modelStatus.textContent =
    "Loading...";


  anchorStatus.textContent =
    "Hidden";


  effectStatus.textContent =
    "Loading...";


  captureStatus.textContent =
    "Ready";


  errorStatus.textContent =
    "None";


  video.style.display =
    "none";


  threeLayer.style.display =
    "none";


  frontThreeLayer.style.display =
    "none";


  effectOverlay.style.display =
    "none";


  overlay.style.display =
    "none";


  startButton.disabled =
    true;


  stopButton.disabled =
    true;


  switchButton.disabled =
    true;


  captureButton.disabled =
    true;
}


/* =========================================================
   APPLICATION BOOT
========================================================= */

async function initializeApplication() {

  try {

    console.log(
      "[Human AR] M8.12B — Per-effect Layering + Orbit Transition"
    );


    initializeUI();


    /*
     * Build the model checkboxes directly from MODEL_REGISTRY.
     */

    createModelToggles();


    bindModelToggleEvents();


    /*
     * Stars remain a BODY_EFFECT rather than a GLB registry model.
     */

    initializeStars();


    /*
     * Three.js is synchronous.
     * Both BACK and FRONT renderers are created here.
     */

    initializeThree();


    /*
     * MediaPipe and GLBs can load independently.
     */

    await Promise.all([

      initializeMediaPipe(),

      loadAllModels()
    ]);


    updateControls();


    if (
      systemReady()
    ) {

      threeStatus.textContent =
        `Ready r${THREE.REVISION}`;


      startButton.disabled =
        false;


      switchButton.disabled =
        false;


      captureStatus.textContent =
        "Ready";


      console.log(
        "[Human AR] System Ready"
      );

    } else {

      throw new Error(
        "Human AR initialization incomplete"
      );
    }


  } catch (
  error
  ) {

    startButton.disabled =
      true;


    captureButton.disabled =
      true;


    setError(
      error
    );


    console.error(
      "[Human AR] Initialization failed:",
      error
    );
  }
}


/* =========================================================
   START APPLICATION
========================================================= */

initializeApplication();