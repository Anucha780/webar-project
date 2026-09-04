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

const overlay =
  document.querySelector("#pose-overlay");

const effectOverlay =
  document.querySelector("#effect-overlay");

const effectCtx =
  effectOverlay.getContext("2d");

const starsToggle =
  document.querySelector("#toggle-stars");

const ctx =
  overlay.getContext("2d");

const placeholder =
  document.querySelector("#camera-placeholder");


const butterflyToggle =
  document.querySelector("#toggle-butterfly");

const waveboyToggle =
  document.querySelector("#toggle-waveboy");


const startButton =
  document.querySelector("#start-camera");

const switchButton =
  document.querySelector("#switch-camera");

const captureButton =
  document.querySelector("#capture-photo");

const stopButton =
  document.querySelector("#stop-camera");


const statusElement =
  document.querySelector("#status");

const mediapipeStatus =
  document.querySelector("#mediapipe-status");

const threeStatus =
  document.querySelector("#three-status");

const activeModelStatus =
  document.querySelector("#active-model-status");

const behaviorStatus =
  document.querySelector("#behavior-status");

const glbPathStatus =
  document.querySelector("#glb-path-status");

const glbLoadStatus =
  document.querySelector("#glb-load-status");

const meshStatus =
  document.querySelector("#mesh-status");

const materialStatus =
  document.querySelector("#material-status");

const bboxStatus =
  document.querySelector("#bbox-status");

const sizeStatus =
  document.querySelector("#size-status");

const animationStatus =
  document.querySelector("#animation-status");

const clipStatus =
  document.querySelector("#clip-status");

const selectedClipStatus =
  document.querySelector("#selected-clip-status");

const cameraStatus =
  document.querySelector("#camera-status");

const personStatus =
  document.querySelector("#person-status");

const anchorStatus =
  document.querySelector("#anchor-status");

const captureStatus =
  document.querySelector("#capture-status");

const errorStatus =
  document.querySelector("#error-status");


/* =========================================================
   CAMERA / MEDIAPIPE
========================================================= */

let stream =
  null;

let facingMode =
  "user";

let poseLandmarker =
  null;

let imageSegmenter =
  null;

let cameraRunning =
  false;

let animationFrameId =
  null;

let lastVideoTime =
  -1;

let lastFrameTimestamp =
  performance.now();

let latestLandmarks =
  null;


/* =========================================================
   MULTI MODEL
========================================================= */

const modelInstances =
  new Map();

let allModelsReady =
  false;


const enabledEffects = {
  butterfly: true,
  waveboy: true,
  stars: true
};

/* =========================================================
   BODY EFFECT — STARS
========================================================= */

const starParticles =
  [];

const STAR_COUNT =
  14;

let starsInitialized =
  false;

function initializeStars() {

  starParticles.length =
    0;


  for (
    let i = 0;
    i < STAR_COUNT;
    i++
  ) {

    starParticles.push({

      angle:
        (
          i /
          STAR_COUNT
        ) *
        Math.PI *
        2,

      speed:
        0.45 +
        Math.random() *
        0.35,

      radiusMultiplier:
        0.9 +
        Math.random() *
        0.45,

      verticalOffset:
        -0.45 +
        Math.random() *
        0.9,

      size:
        5 +
        Math.random() *
        6,

      phase:
        Math.random() *
        Math.PI *
        2
    });
  }


  starsInitialized =
    true;
}

function drawStarShape(
  x,
  y,
  outerRadius,
  innerRadius
) {

  const spikes =
    5;


  let rotation =
    -Math.PI / 2;


  const step =
    Math.PI /
    spikes;


  effectCtx.beginPath();


  effectCtx.moveTo(
    x +
    Math.cos(
      rotation
    ) *
    outerRadius,

    y +
    Math.sin(
      rotation
    ) *
    outerRadius
  );


  for (
    let i = 0;
    i < spikes;
    i++
  ) {

    effectCtx.lineTo(

      x +
      Math.cos(
        rotation
      ) *
      outerRadius,

      y +
      Math.sin(
        rotation
      ) *
      outerRadius
    );


    rotation +=
      step;


    effectCtx.lineTo(

      x +
      Math.cos(
        rotation
      ) *
      innerRadius,

      y +
      Math.sin(
        rotation
      ) *
      innerRadius
    );


    rotation +=
      step;
  }


  effectCtx.closePath();


  effectCtx.fillStyle =
    "rgba(255,255,255,0.95)";


  effectCtx.fill();
}

function drawStarsEffect(
  timestamp
) {

  effectCtx.clearRect(
    0,
    0,
    effectOverlay.width,
    effectOverlay.height
  );


  if (
    !enabledEffects.stars ||
    !trackedBody.valid
  ) {

    return;
  }


  if (
    !starsInitialized
  ) {

    initializeStars();
  }


  const centerX =
    trackedBody.centerX *
    effectOverlay.width;


  const centerY =
    (
      1 -
      trackedBody.centerY
    ) *
    effectOverlay.height;


  const bodyWidth =
    trackedBody.shoulderWidth *
    effectOverlay.width;


  const bodyHeight =
    trackedBody.torsoHeight *
    effectOverlay.height;


  for (
    const star
    of starParticles
  ) {

    const time =
      timestamp /
      1000;


    const angle =
      star.angle +
      time *
      star.speed;


    const radiusX =
      bodyWidth *
      1.3 *
      star.radiusMultiplier;


    const radiusY =
      bodyHeight *
      0.9 *
      star.radiusMultiplier;


    const x =
      centerX +
      Math.cos(
        angle
      ) *
      radiusX;


    const y =
      centerY +
      star.verticalOffset *
      bodyHeight +
      Math.sin(
        angle * 1.5 +
        star.phase
      ) *
      radiusY *
      0.35;


    const pulse =
      0.8 +
      Math.sin(
        time * 3 +
        star.phase
      ) *
      0.2;


    const size =
      star.size *
      pulse *
      (
        window.devicePixelRatio ||
        1
      );


    drawStarShape(
      x,
      y,
      size,
      size * 0.45
    );
  }
}



/* =========================================================
   SEGMENTATION
========================================================= */

let segmentationBusy =
  false;

let lastSegmentationTime =
  0;

let segmentationMaskReady =
  false;

let previousMaskValues =
  null;


const SEGMENTATION_INTERVAL_MS =
  1000 / 20;

const MASK_CONFIDENCE_LOW =
  0.12;

const MASK_CONFIDENCE_HIGH =
  0.52;

const MASK_EXPANSION =
  1.02;

const MASK_TEMPORAL_ALPHA =
  0.70;


/* =========================================================
   OFFSCREEN CANVASES
========================================================= */

const maskCanvas =
  document.createElement("canvas");

const maskCtx =
  maskCanvas.getContext(
    "2d",
    {
      willReadFrequently:
        true
    }
  );


const compositeCanvas =
  document.createElement("canvas");

const compositeCtx =
  compositeCanvas.getContext("2d");


const captureCanvas =
  document.createElement("canvas");

const captureCtx =
  captureCanvas.getContext("2d");


/* =========================================================
   THREE
========================================================= */

let scene =
  null;

let threeCamera =
  null;

let renderer =
  null;


/* =========================================================
   ORBIT SETTINGS
========================================================= */

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

const FOLLOW_ORBIT_DIRECTION =
  true;


let orbitAngle =
  0;

let orbitDepth =
  0;


/* =========================================================
   SMOOTHING
========================================================= */

const POSITION_SMOOTHING =
  12;

const SCALE_SMOOTHING =
  9;

const ROTATION_SMOOTHING =
  8;

const MIN_VISIBILITY =
  0.55;


/* =========================================================
   LANDMARK INDEX
========================================================= */

const LANDMARK = {

  NOSE:
    0,

  LEFT_SHOULDER:
    11,

  RIGHT_SHOULDER:
    12,

  LEFT_HIP:
    23,

  RIGHT_HIP:
    24
};


/* =========================================================
   TRACKED BODY
========================================================= */

const trackedBody = {

  valid:
    false,

  centerX:
    0.5,

  centerY:
    0.5,

  headX:
    0.5,

  headY:
    0.25,
     
  shoulderWidth:
    0.2,

  torsoHeight:
    0.2,

  leftShoulderX:
    0.4,

  leftShoulderY:
    0.4,

  rightShoulderX:
    0.6,

  rightShoulderY:
    0.4
};


/* =========================================================
   DEBUG
========================================================= */

function setStatus(
  message
) {

  if (
    statusElement
  ) {

    statusElement.textContent =
      message;
  }

  console.log(
    `[Human AR] ${message}`
  );
}


function clearError() {

  if (
    errorStatus
  ) {

    errorStatus.textContent =
      "None";
  }
}


function setError(
  error
) {

  console.error(
    "[Human AR Error]",
    error
  );

  if (
    errorStatus
  ) {

    errorStatus.textContent =
      `${error.name}: ${error.message}`;
  }
}


/* =========================================================
   EFFECT STATE
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


function updateEffectDebug() {

  const names =
    [];

  const behaviors =
    [];


  if (
    enabledEffects.butterfly
  ) {

    names.push(
      "Butterfly"
    );

    behaviors.push(
      "ORBIT"
    );
  }


  if (
    enabledEffects.waveboy
  ) {

    names.push(
      "Waveboy"
    );

    behaviors.push(
      "SHOULDER"
    );
  }

  if (
  enabledEffects.stars
) {

  names.push(
    "Stars"
  );

  behaviors.push(
    "BODY_EFFECT"
  );
}

  activeModelStatus.textContent =
    names.length > 0
      ? names.join(" + ")
      : "None";


  behaviorStatus.textContent =
    behaviors.length > 0
      ? behaviors.join(" + ")
      : "None";
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


function getCaptureEffectName() {

  const names =
    [];


  if (
    enabledEffects.butterfly
  ) {

    names.push(
      "butterfly"
    );
  }


  if (
    enabledEffects.waveboy
  ) {

    names.push(
      "waveboy"
    );
  }

  if (
  enabledEffects.stars
) {

  names.push(
    "stars"
  );
}

  if (
    names.length === 0
  ) {

    return "no-effects";
  }


  return names.join("-");
}


/* =========================================================
   DAMP
========================================================= */

function damp(
  current,
  target,
  smoothing,
  delta
) {

  const factor =
    1 -
    Math.exp(
      -smoothing *
      delta
    );

  return THREE.MathUtils.lerp(
    current,
    target,
    factor
  );
}


function dampAngle(
  current,
  target,
  smoothing,
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


  const factor =
    1 -
    Math.exp(
      -smoothing *
      delta
    );


  return (
    current +
    difference *
    factor
  );
}


/* =========================================================
   THREE INITIALIZE
========================================================= */

function initializeThree() {

  threeStatus.textContent =
    "Initializing...";


  scene =
    new THREE.Scene();


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

      alpha:
        true,

      antialias:
        true,

      preserveDrawingBuffer:
        true
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


  renderer.outputColorSpace =
    THREE.SRGBColorSpace;


  threeLayer.appendChild(
    renderer.domElement
  );


  const ambient =
    new THREE.AmbientLight(
      0xffffff,
      2
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


  threeStatus.textContent =
    `Ready r${THREE.REVISION}`;
}


/* =========================================================
   TORSO TEST OBJECT
========================================================= */



/* =========================================================
   HEAD TEST OBJECT
========================================================= */



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
   LOAD ALL
========================================================= */

async function loadAllModels() {

  allModelsReady =
    false;


  glbLoadStatus.textContent =
    "Loading...";


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


    clearError();


    setStatus(
      "All models ready"
    );


    updateControls();

  } catch (error) {

    allModelsReady =
      false;


    glbLoadStatus.textContent =
      "FAILED";


    setError(
      error
    );


    setStatus(
      "Model load failed"
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


  glbPathStatus.textContent =
    instances
      .map(
        item =>
          item.config.path
      )
      .join(" | ");


  glbLoadStatus.textContent =
    `PASS ${instances.length}/${MODEL_REGISTRY.length}`;


  meshStatus.textContent =
    String(
      totalMeshes
    );


  materialStatus.textContent =
    String(
      totalMaterials
    );


  bboxStatus.textContent =
    "Valid";


  sizeStatus.textContent =
    instances
      .map(
        item => {

          const s =
            item.stats.size;

          return `${item.config.name}: ${s.x.toFixed(3)}×${s.y.toFixed(3)}×${s.z.toFixed(3)}`;
        }
      )
      .join(" | ");


  animationStatus.textContent =
    String(
      totalAnimations
    );


  clipStatus.textContent =
    instances
      .map(
        item =>
          `${item.config.name}: ${
            item.stats.clipNames.length
              ? item.stats.clipNames.join(", ")
              : "None"
          }`
      )
      .join(" | ");


  selectedClipStatus.textContent =
    instances
      .map(
        item =>
          `${item.config.name}: ${item.stats.selectedClipName}`
      )
      .join(" | ");


  updateEffectDebug();
}


/* =========================================================
   READY / CONTROLS
========================================================= */

function systemReady() {

  return Boolean(
    poseLandmarker &&
    imageSegmenter &&
    renderer &&
    allModelsReady
  );
}


function updateControls() {

  const ready =
    systemReady();


  startButton.disabled =
    !ready ||
    cameraRunning;


  butterflyToggle.disabled =
    !allModelsReady;


  waveboyToggle.disabled =
    !allModelsReady;

  starsToggle.disabled =
    !allModelsReady;

  switchButton.disabled =
    !cameraRunning;


  captureButton.disabled =
    !cameraRunning;


  stopButton.disabled =
    !cameraRunning;


  if (
    ready &&
    !cameraRunning
  ) {

    setStatus(
      "Ready — Start Camera"
    );
  }
}


/* =========================================================
   MEDIAPIPE
========================================================= */

async function initializeMediaPipe() {

  try {

    setStatus(
      "Loading MediaPipe..."
    );


    mediapipeStatus.textContent =
      "Loading WASM...";


    const vision =
      await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
      );


    mediapipeStatus.textContent =
      "Loading Pose...";


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


    mediapipeStatus.textContent =
      "Loading Segmenter...";


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

          outputConfidenceMasks:
            true,

          outputCategoryMask:
            false
        }
      );


    mediapipeStatus.textContent =
      "Pose Ready | Seg Ready";


    clearError();


    updateControls();

  } catch (error) {

    mediapipeStatus.textContent =
      "Failed";


    setError(
      error
    );


    setStatus(
      "MediaPipe initialization failed"
    );
  }
}


/* =========================================================
   CAMERA
========================================================= */

function updateCameraDisplay() {

  video.style.transform =
    facingMode === "user"
      ? "scaleX(-1)"
      : "none";
}


async function startCamera() {

  if (
    !systemReady()
  ) {

    setStatus(
      "System is not ready"
    );

    return;
  }


  clearError();


  try {

    if (
      stream
    ) {

      stopStream();
    }


    stream =
      await navigator.mediaDevices
        .getUserMedia({

          audio:
            false,

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


    updateCameraDisplay();


    video.style.display =
      "block";


    threeLayer.style.display =
      "block";

    effectOverlay.style.display =
      "block";

    overlay.style.display =
      "block";


    placeholder.style.display =
      "none";


    cameraRunning =
      true;


    cameraStatus.textContent =
      "Running";


    captureStatus.textContent =
      "Ready";


    resizeOverlay();


    resizeThree();


    lastVideoTime =
      -1;


    lastFrameTimestamp =
      performance.now();


    lastSegmentationTime =
      0;


    segmentationBusy =
      false;


    segmentationMaskReady =
      false;


    previousMaskValues =
      null;


    latestLandmarks =
      null;


    orbitAngle =
      0;


    orbitDepth =
      0;


    trackedBody.valid =
      false;


    hideAllModels();


    clearError();


    setStatus(
      "Human AR running"
    );


    updateControls();


    predictPose();

  } catch (error) {

    cameraRunning =
      false;


    setError(
      error
    );


    setStatus(
      "Camera failed"
    );


    updateControls();
  }
}


/* =========================================================
   STOP / SWITCH
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


function stopStream() {

  cameraRunning =
    false;


  if (
    animationFrameId !==
    null
  ) {

    cancelAnimationFrame(
      animationFrameId
    );


    animationFrameId =
      null;
  }


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


  video.srcObject =
    null;


  video.style.display =
    "none";


  threeLayer.style.display =
    "none";

  effectOverlay.style.display =
    "none";  

  overlay.style.display =
    "none";


  trackedBody.valid =
    false;


  latestLandmarks =
    null;


  segmentationMaskReady =
    false;


  segmentationBusy =
    false;


  previousMaskValues =
    null;


  orbitDepth =
    0;


  hideAllModels();


  clearOverlay();
  
  effectCtx.clearRect(
    0,
    0,
    effectOverlay.width,
    effectOverlay.height
  );

  cameraStatus.textContent =
    "Stopped";


  personStatus.textContent =
    "Not detected";


  anchorStatus.textContent =
    "Hidden";


  captureStatus.textContent =
    "Camera stopped";


  placeholder.style.display =
    "flex";


  placeholder.textContent =
    "Camera stopped";


  updateControls();
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


  stopStream();


  await startCamera();
}


/* =========================================================
   RESIZE
========================================================= */

function resizeOverlay() {

  const rect =
    overlay.getBoundingClientRect();


  const dpr =
    window.devicePixelRatio || 1;


  overlay.width =
    Math.round(
      rect.width *
      dpr
    );


  overlay.height =
    Math.round(
      rect.height *
      dpr
    );

  effectOverlay.width =
    overlay.width;

  effectOverlay.height =
    overlay.height;

  compositeCanvas.width =
    overlay.width;


  compositeCanvas.height =
    overlay.height;


  captureCanvas.width =
    overlay.width;


  captureCanvas.height =
    overlay.height;
}


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


function clearOverlay() {

  ctx.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );
}


/* =========================================================
   COVER TRANSFORM
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
    sourceWidth *
    scale;


  const renderedHeight =
    sourceHeight *
    scale;


  return {

    sourceWidth,

    sourceHeight,

    displayWidth,

    displayHeight,

    scale,

    renderedWidth,

    renderedHeight,

    cropX:
      (
        renderedWidth -
        displayWidth
      ) / 2,

    cropY:
      (
        renderedHeight -
        displayHeight
      ) / 2
  };
}


/* =========================================================
   LANDMARK MAP
========================================================= */

function landmarkToCanvas(
  landmark
) {

  const transform =
    getCoverTransform();


  if (
    !transform
  ) {

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


  return {

    x:
      sourceX *
      transform.scale -
      transform.cropX,

    y:
      sourceY *
      transform.scale -
      transform.cropY
  };
}


function canvasToThree(
  point
) {

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
   UTILITY
========================================================= */

function distance2D(
  a,
  b
) {

  const dx =
    a.x -
    b.x;


  const dy =
    a.y -
    b.y;


  return Math.sqrt(
    dx * dx +
    dy * dy
  );
}


function landmarkReliable(
  landmark
) {

  if (
    !landmark
  ) {

    return false;
  }


  if (
    landmark.visibility ===
    undefined
  ) {

    return true;
  }


  return (
    landmark.visibility >=
    MIN_VISIBILITY
  );
}


function smoothStep(
  edge0,
  edge1,
  value
) {

  let t =
    (
      value -
      edge0
    ) /
    (
      edge1 -
      edge0
    );


  t =
    THREE.MathUtils.clamp(
      t,
      0,
      1
    );


  return (
    t *
    t *
    (
      3 -
      2 * t
    )
  );
}


/* =========================================================
   BODY TRACK
========================================================= */

function updateTrackedBody(
  landmarks
) {

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


  const reliable =
    landmarkReliable(
      leftShoulder
    ) &&
    landmarkReliable(
      rightShoulder
    ) &&
    landmarkReliable(
      leftHip
    ) &&
    landmarkReliable(
      rightHip
    );


  if (
    !reliable
  ) {

    trackedBody.valid =
      false;

    return;
  }


  const ls =
    landmarkToCanvas(
      leftShoulder
    );


  const rs =
    landmarkToCanvas(
      rightShoulder
    );


  const lh =
    landmarkToCanvas(
      leftHip
    );


  const rh =
    landmarkToCanvas(
      rightHip
    );


  const shoulderCenter = {

    x:
      (
        ls.x +
        rs.x
      ) / 2,

    y:
      (
        ls.y +
        rs.y
      ) / 2
  };


  const hipCenter = {

    x:
      (
        lh.x +
        rh.x
      ) / 2,

    y:
      (
        lh.y +
        rh.y
      ) / 2
  };


  const torsoCanvas = {

    x:
      (
        shoulderCenter.x +
        hipCenter.x
      ) / 2,

    y:
      (
        shoulderCenter.y +
        hipCenter.y
      ) / 2
  };


  const torsoThree =
    canvasToThree(
      torsoCanvas
    );


  const leftShoulderThree =
    canvasToThree(
      ls
    );


  const rightShoulderThree =
    canvasToThree(
      rs
    );
  
  const noseCanvas =
    landmarkToCanvas(
     nose
    );


  const noseThree =
    canvasToThree(
      noseCanvas
    );


  trackedBody.centerX =
    torsoThree.x;


  trackedBody.centerY =
    torsoThree.y;


  trackedBody.shoulderWidth =
    distance2D(
      ls,
      rs
    ) /
    overlay.width;


  trackedBody.torsoHeight =
    distance2D(
      shoulderCenter,
      hipCenter
    ) /
    overlay.height;


  trackedBody.leftShoulderX =
    leftShoulderThree.x;


  trackedBody.leftShoulderY =
    leftShoulderThree.y;


  trackedBody.rightShoulderX =
    rightShoulderThree.x;


  trackedBody.rightShoulderY =
    rightShoulderThree.y;

  if (
    landmarkReliable(
      nose
   )
  ) {

  trackedBody.headX =
    noseThree.x;

  trackedBody.headY =
    noseThree.y;
}
  trackedBody.valid =
    true;

}


/* =========================================================
   SEGMENTATION
========================================================= */

function requestSegmentation(
  timestamp
) {

  if (
    !imageSegmenter ||
    segmentationBusy ||
    !trackedBody.valid ||
    video.readyState < 2
  ) {

    return;
  }


  if (
    timestamp -
    lastSegmentationTime <
    SEGMENTATION_INTERVAL_MS
  ) {

    return;
  }


  lastSegmentationTime =
    timestamp;


  segmentationBusy =
    true;


  try {

    imageSegmenter.segmentForVideo(
      video,
      timestamp,
      handleSegmentationResult
    );

  } catch (error) {

    segmentationBusy =
      false;


    setError(
      error
    );
  }
}


function handleSegmentationResult(
  result
) {

  try {

    if (
      !result ||
      !result.confidenceMasks ||
      result.confidenceMasks.length ===
        0
    ) {

      segmentationMaskReady =
        false;

      return;
    }


    const mask =
      result.confidenceMasks[0];


    const width =
      mask.width;


    const height =
      mask.height;


    const currentValues =
      mask.getAsFloat32Array();


    if (
      !previousMaskValues ||
      previousMaskValues.length !==
        currentValues.length
    ) {

      previousMaskValues =
        new Float32Array(
          currentValues.length
        );


      previousMaskValues.set(
        currentValues
      );

    } else {

      for (
        let i = 0;
        i < currentValues.length;
        i++
      ) {

        previousMaskValues[i] =
          previousMaskValues[i] *
          (
            1 -
            MASK_TEMPORAL_ALPHA
          )
          +
          currentValues[i] *
          MASK_TEMPORAL_ALPHA;
      }
    }


    if (
      maskCanvas.width !==
        width ||
      maskCanvas.height !==
        height
    ) {

      maskCanvas.width =
        width;


      maskCanvas.height =
        height;
    }


    const rgba =
      new Uint8ClampedArray(
        width *
        height *
        4
      );


    let targetIndex =
      0;


    for (
      let i = 0;
      i < previousMaskValues.length;
      i++
    ) {

      const confidence =
        previousMaskValues[i];


      const alpha =
        smoothStep(
          MASK_CONFIDENCE_LOW,
          MASK_CONFIDENCE_HIGH,
          confidence
        ) *
        255;


      rgba[
        targetIndex
      ] =
        255;


      rgba[
        targetIndex + 1
      ] =
        255;


      rgba[
        targetIndex + 2
      ] =
        255;


      rgba[
        targetIndex + 3
      ] =
        alpha;


      targetIndex +=
        4;
    }


    maskCtx.putImageData(

      new ImageData(
        rgba,
        width,
        height
      ),

      0,
      0
    );


    segmentationMaskReady =
      true;


    if (
      typeof mask.close ===
      "function"
    ) {

      mask.close();
    }

  } catch (error) {

    segmentationMaskReady =
      false;


    setError(
      error
    );

  } finally {

    segmentationBusy =
      false;
  }
}


/* =========================================================
   DRAW COVER
========================================================= */

function drawCoverSource(
  targetContext,
  source,
  transform,
  expansion = 1,
  mirror = true
) {

  targetContext.save();


  const extraWidth =
    transform.renderedWidth *
    (
      expansion -
      1
    );


  const extraHeight =
    transform.renderedHeight *
    (
      expansion -
      1
    );


  const drawWidth =
    transform.renderedWidth +
    extraWidth;


  const drawHeight =
    transform.renderedHeight +
    extraHeight;


  const drawX =
    -transform.cropX -
    extraWidth / 2;


  const drawY =
    -transform.cropY -
    extraHeight / 2;


  if (
    mirror &&
    facingMode === "user"
  ) {

    targetContext.translate(
      transform.displayWidth,
      0
    );


    targetContext.scale(
      -1,
      1
    );
  }


  targetContext.drawImage(
    source,
    drawX,
    drawY,
    drawWidth,
    drawHeight
  );


  targetContext.restore();
}


/* =========================================================
   OCCLUSION
========================================================= */

function buildHumanOcclusionLayer() {

  if (
    !segmentationMaskReady
  ) {

    return false;
  }


  const transform =
    getCoverTransform();


  if (
    !transform
  ) {

    return false;
  }


  compositeCtx.clearRect(
    0,
    0,
    compositeCanvas.width,
    compositeCanvas.height
  );


  compositeCtx.globalCompositeOperation =
    "source-over";


  drawCoverSource(
    compositeCtx,
    video,
    transform,
    1,
    true
  );


  compositeCtx.globalCompositeOperation =
    "destination-in";


  drawCoverSource(
    compositeCtx,
    maskCanvas,
    transform,
    MASK_EXPANSION,
    true
  );


  compositeCtx.globalCompositeOperation =
    "source-over";


  return true;
}


function shouldUseHumanOcclusion() {

  return (
    effectIsEnabled(
      "butterfly"
    ) &&
    orbitDepth < 0
  );
}


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


  const radiusX =
    trackedBody.shoulderWidth *
    ORBIT_RADIUS_X;


  const radiusY =
    trackedBody.torsoHeight *
    ORBIT_RADIUS_Y;


  const centerYOffset =
    trackedBody.torsoHeight *
    ORBIT_CENTER_Y;


  const targetX =
    trackedBody.centerX +
    orbitX *
    radiusX;


  const targetY =
    trackedBody.centerY +
    centerYOffset +
    orbitY *
    radiusY;


  const bodyReference =
    trackedBody.shoulderWidth *
    0.65
    +
    trackedBody.torsoHeight *
    0.35;


  const baseScale =
    bodyReference *
    config.scaleMultiplier;


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
    orbitDepth *
    0.1;


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


  const depthText =
    orbitDepth >= 0
      ? "FRONT"
      : "BACK";


  return (
    `Butterfly ORBIT ${depthText}`
  );
}


/* =========================================================
   SHOULDER
========================================================= */

function updateShoulderBehavior(
  instance,
  delta
) {

  const config =
    instance.config;


  const shoulderConfig =
    config.shoulder || {};


  const requestedSide =
    shoulderConfig.side === "left"
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
    requestedSide === "left"
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
    `Waveboy SHOULDER ${requestedSide.toUpperCase()}`
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
    config.torso || {};


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
    config.head || {};


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
    config.beside || {};


  const side =
    beside.side === "left"
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
      instance.config;


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
      ? debug.join(" | ")
      : "No effects";
}


/* =========================================================
   ANIMATION
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
   POSE DEBUG
========================================================= */

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
    2 *
    dpr;


  ctx.stroke();
}


function drawPoint(
  landmark
) {

  const point =
    landmarkToCanvas(
      landmark
    );


  const dpr =
    window.devicePixelRatio || 1;


  ctx.beginPath();


  ctx.arc(
    point.x,
    point.y,
    4 * dpr,
    0,
    Math.PI * 2
  );


  ctx.fillStyle =
    "#00ff88";


  ctx.fill();
}


function drawPoseDebug(
  landmarks
) {

  const ls =
    landmarks[
      LANDMARK.LEFT_SHOULDER
    ];


  const rs =
    landmarks[
      LANDMARK.RIGHT_SHOULDER
    ];


  const lh =
    landmarks[
      LANDMARK.LEFT_HIP
    ];


  const rh =
    landmarks[
      LANDMARK.RIGHT_HIP
    ];


  drawLine(
    ls,
    rs
  );


  drawLine(
    ls,
    lh
  );


  drawLine(
    rs,
    rh
  );


  drawLine(
    lh,
    rh
  );


  drawPoint(
    ls
  );


  drawPoint(
    rs
  );


  drawPoint(
    lh
  );


  drawPoint(
    rh
  );
}


/* =========================================================
   OVERLAY
========================================================= */

function drawOverlay() {

  clearOverlay();


  if (
    !latestLandmarks ||
    !trackedBody.valid
  ) {

    return;
  }


  drawFullHumanOcclusion();


  drawPoseDebug(
    latestLandmarks
  );
}


/* =========================================================
   CAPTURE
========================================================= */

async function capturePhoto() {

  if (
    !cameraRunning ||
    !renderer ||
    video.readyState < 2
  ) {

    captureStatus.textContent =
      "Camera not ready";


    return;
  }


  captureButton.disabled =
    true;


  captureStatus.textContent =
    "Capturing...";


  try {

    const transform =
      getCoverTransform();


    if (
      !transform
    ) {

      throw new Error(
        "Capture transform unavailable"
      );
    }


    captureCanvas.width =
      overlay.width;


    captureCanvas.height =
      overlay.height;


    captureCtx.clearRect(
      0,
      0,
      captureCanvas.width,
      captureCanvas.height
    );


    captureCtx.globalCompositeOperation =
      "source-over";


    drawCoverSource(
      captureCtx,
      video,
      transform,
      1,
      true
    );

    drawStarsEffect(
      performance.now()
    );

    renderer.render(
      scene,
      threeCamera
    );


    captureCtx.drawImage(
      renderer.domElement,
      0,
      0,
      captureCanvas.width,
      captureCanvas.height
    );

    captureCtx.drawImage(
      effectOverlay,
      0,
      0,
      captureCanvas.width,
      captureCanvas.height
    );


    captureCtx.drawImage(
      effectOverlay,
      0,
      0,
      captureCanvas.width,
      captureCanvas.height
    );

    if (
      shouldUseHumanOcclusion() &&
      segmentationMaskReady
    ) {

      if (
        buildHumanOcclusionLayer()
      ) {

        captureCtx.drawImage(
          compositeCanvas,
          0,
          0,
          captureCanvas.width,
          captureCanvas.height
        );
      }
    }


    const blob =
      await new Promise(
        resolve => {

          captureCanvas.toBlob(
            resolve,
            "image/jpeg",
            0.95
          );
        }
      );


    if (
      !blob
    ) {

      throw new Error(
        "Unable to create capture"
      );
    }


    const url =
      URL.createObjectURL(
        blob
      );


    const timestamp =
      new Date()
        .toISOString()
        .replace(
          /[:.]/g,
          "-"
        );


    const link =
      document.createElement(
        "a"
      );


    link.href =
      url;


    link.download =
      `human-ar-${getCaptureEffectName()}-${timestamp}.jpg`;


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
      3000
    );


    captureStatus.textContent =
      "PASS — Photo captured";


    clearError();

  } catch (error) {

    captureStatus.textContent =
      "FAILED";


    setError(
      error
    );

  } finally {

    if (
      cameraRunning
    ) {

      captureButton.disabled =
        false;
    }
  }
}


/* =========================================================
   MAIN LOOP
========================================================= */

function predictPose() {

  if (
    !cameraRunning ||
    !poseLandmarker
  ) {

    return;
  }


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


  updateModelAnimations(
    delta
  );


  if (
    video.readyState >= 2 &&
    video.currentTime !==
      lastVideoTime
  ) {

    lastVideoTime =
      video.currentTime;


    try {

      const result =
        poseLandmarker.detectForVideo(
          video,
          now
        );


      if (
        result.landmarks &&
        result.landmarks.length > 0
      ) {

        personStatus.textContent =
          "Detected";


        latestLandmarks =
          result.landmarks[0];


        updateTrackedBody(
          latestLandmarks
        );


        requestSegmentation(
          now
        );

      } else {

        personStatus.textContent =
          "Not detected";


        trackedBody.valid =
          false;


        latestLandmarks =
          null;


        segmentationMaskReady =
          false;


        previousMaskValues =
          null;


        hideAllModels();


        anchorStatus.textContent =
          "Hidden";
      }

    } catch (error) {

      hideAllModels();


      anchorStatus.textContent =
        "Pose Error";


      setError(
        error
      );
    }
  }


  try {

    updateAllModelBehaviors(
      delta
    );


  } catch (error) {

    hideAllModels();


    anchorStatus.textContent =
      "Behavior Error";


    setError(
      error
    );
  }


  renderer.render(
    scene,
    threeCamera
  );


  drawStarsEffect(
    now
  );
  
  drawOverlay();


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


captureButton.addEventListener(
  "click",
  capturePhoto
);


stopButton.addEventListener(
  "click",
  stopCamera
);


butterflyToggle.addEventListener(
  "change",
  () => {

    enabledEffects.butterfly =
      butterflyToggle.checked;


    updateEffectVisibility();


    updateEffectDebug();


    clearError();


    setStatus(
      enabledEffects.butterfly
        ? "Butterfly enabled"
        : "Butterfly disabled"
    );
  }
);


waveboyToggle.addEventListener(
  "change",
  () => {

    enabledEffects.waveboy =
      waveboyToggle.checked;


    updateEffectVisibility();


    updateEffectDebug();


    clearError();


    setStatus(
      enabledEffects.waveboy
        ? "Waveboy enabled"
        : "Waveboy disabled"
    );
  }
);

starsToggle.addEventListener(
  "change",
  () => {

    enabledEffects.stars =
      starsToggle.checked;


    if (
      !enabledEffects.stars
    ) {

      effectCtx.clearRect(
        0,
        0,
        effectOverlay.width,
        effectOverlay.height
      );
    }


    updateEffectDebug();


    clearError();


    setStatus(
      enabledEffects.stars
        ? "Stars enabled"
        : "Stars disabled"
    );
  }
);

video.addEventListener(
  "loadedmetadata",
  () => {

    resizeOverlay();


    resizeThree();
  }
);


window.addEventListener(
  "resize",
  () => {

    resizeOverlay();


    resizeThree();
  }
);


/* =========================================================
   INITIALIZATION
========================================================= */

console.log(
  "[Human AR] Milestone 8.6B — Effect Toggles"
);


startButton.disabled =
  true;


switchButton.disabled =
  true;


captureButton.disabled =
  true;


stopButton.disabled =
  true;


butterflyToggle.disabled =
  true;


waveboyToggle.disabled =
  true;

starsToggle.disabled =
  true;

butterflyToggle.checked =
  enabledEffects.butterfly;


waveboyToggle.checked =
  enabledEffects.waveboy;

starsToggle.checked =
  enabledEffects.stars;

updateEffectDebug();


try {

  initializeThree();

  loadAllModels();

} catch (error) {

  setError(
    error
  );


  setStatus(
    "Three.js initialization failed"
  );
}


initializeMediaPipe();