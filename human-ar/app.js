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
  getModelConfig
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

const ctx =
  overlay.getContext("2d");

const placeholder =
  document.querySelector("#camera-placeholder");


const modelSelector =
  document.querySelector("#model-selector");

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
   STATE
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
   MODEL STATE
========================================================= */

let activeModelId =
  DEFAULT_MODEL_ID;

let activeModelConfig =
  null;

let loadedModel =
  null;

let mixer =
  null;

let activeAction =
  null;


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
   SEGMENTATION CANVASES
========================================================= */

const maskCanvas =
  document.createElement("canvas");

const maskCtx =
  maskCanvas.getContext(
    "2d",
    {
      willReadFrequently: true
    }
  );


const compositeCanvas =
  document.createElement("canvas");

const compositeCtx =
  compositeCanvas.getContext("2d");


/* =========================================================
   CAPTURE CANVAS
========================================================= */

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

let modelAnchor =
  null;


/* =========================================================
   ORBIT
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
   LANDMARKS
========================================================= */

const LANDMARK = {

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
   TRACKING STATE
========================================================= */

let orbitAngle =
  0;

let orbitDepth =
  0;


const trackedBody = {

    valid:
    false,

  centerX:
    0.5,

  centerY:
    0.5,

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

  statusElement.textContent =
    message;


  console.log(
    `[Human AR] ${message}`
  );
}


function clearError() {

  errorStatus.textContent =
    "None";
}


function setError(
  error
) {

  console.error(
    "[Human AR Error]",
    error
  );


  errorStatus.textContent =
    `${error.name}: ${error.message}`;
}


/* =========================================================
   MODEL SELECTOR
========================================================= */

function initializeModelSelector() {

  modelSelector.innerHTML =
    "";


  for (
    const model
    of MODEL_REGISTRY
  ) {

    const option =
      document.createElement(
        "option"
      );


    option.value =
      model.id;


    option.textContent =
      model.name;


    modelSelector.appendChild(
      option
    );
  }


  modelSelector.value =
    activeModelId;


  modelSelector.disabled =
    MODEL_REGISTRY.length === 0;
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
   THREE INITIALIZATION
========================================================= */

function initializeThree() {

  try {

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


    modelAnchor =
      new THREE.Group();


    modelAnchor.visible =
      false;


    modelAnchor.position.set(
      0.5,
      0.5,
      0
    );


    modelAnchor.scale.setScalar(
      0.1
    );


    scene.add(
      modelAnchor
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

  } catch (error) {

    threeStatus.textContent =
      "Failed";


    setError(
      error
    );


    throw error;
  }
}


/* =========================================================
   MODEL CLEANUP
========================================================= */

function disposeLoadedModel() {

  if (
    mixer
  ) {

    mixer.stopAllAction();
  }


  if (
    loadedModel
  ) {

    loadedModel.traverse(
      object => {

        if (
          !object.isMesh
        ) {

          return;
        }


        if (
          object.geometry
        ) {

          object.geometry.dispose();
        }


        if (
          !object.material
        ) {

          return;
        }


        const materials =
          Array.isArray(
            object.material
          )
            ? object.material
            : [
                object.material
              ];


        for (
          const material
          of materials
        ) {

          for (
            const value
            of Object.values(
              material
            )
          ) {

            if (
              value &&
              value.isTexture
            ) {

              value.dispose();
            }
          }


          material.dispose();
        }
      }
    );


    modelAnchor.remove(
      loadedModel
    );
  }


  loadedModel =
    null;

  mixer =
    null;

  activeAction =
    null;

  modelAnchor.visible =
    false;
}


/* =========================================================
   MODEL DEBUG RESET
========================================================= */

function resetModelDebug() {

  glbLoadStatus.textContent =
    "Loading...";

  meshStatus.textContent =
    "Waiting...";

  materialStatus.textContent =
    "Waiting...";

  bboxStatus.textContent =
    "Waiting...";

  sizeStatus.textContent =
    "Waiting...";

  animationStatus.textContent =
    "Waiting...";

  clipStatus.textContent =
    "Waiting...";

  selectedClipStatus.textContent =
    "None";
}


/* =========================================================
   LOAD MODEL
========================================================= */

function loadActiveModel() {

  const config =
    getModelConfig(
      activeModelId
    );


  if (
    !config
  ) {

    setError(
      new Error(
        `Unknown model ID: ${activeModelId}`
      )
    );

    return;
  }


  activeModelConfig =
    config;


  activeModelStatus.textContent =
    config.name;


  behaviorStatus.textContent =
    config.behavior;


  glbPathStatus.textContent =
    config.path;


  resetModelDebug();


  setStatus(
    `Loading ${config.name}...`
  );


  disposeLoadedModel();


  const loader =
    new GLTFLoader();


  loader.load(

    config.path,


    gltf => {

      try {

        validateAndAttachModel(
          gltf,
          config
        );

      } catch (error) {

        glbLoadStatus.textContent =
          "FAILED";


        setError(
          error
        );


        setStatus(
          `${config.name} validation failed`
        );
      }
    },


    progress => {

      if (
        progress.total >
        0
      ) {

        const percent =
          Math.round(
            (
              progress.loaded /
              progress.total
            ) *
            100
          );


        glbLoadStatus.textContent =
          `${percent}%`;

      } else {

        glbLoadStatus.textContent =
          "Loading...";
      }
    },


    error => {

      console.error(
        "[GLB Loader]",
        error
      );


      glbLoadStatus.textContent =
        "FAILED";


      setError(
        new Error(
          `Unable to load ${config.path}`
        )
      );


      setStatus(
        `${config.name} load failed`
      );
    }
  );
}


/* =========================================================
   GLB VALIDATION
========================================================= */

function validateAndAttachModel(
  gltf,
  config
) {

  if (
    !gltf.scene
  ) {

    throw new Error(
      "GLB has no scene"
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


      if (
        Array.isArray(
          object.material
        )
      ) {

        for (
          const material
          of object.material
        ) {

          materials.add(
            material
          );
        }

      } else {

        materials.add(
          object.material
        );
      }
    }
  );


  meshStatus.textContent =
    String(
      meshCount
    );


  materialStatus.textContent =
    String(
      materials.size
    );


  if (
    meshCount === 0
  ) {

    throw new Error(
      "GLB contains no mesh"
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

    bboxStatus.textContent =
      "EMPTY";


    throw new Error(
      "GLB bounding box is empty"
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


  if (
    !Number.isFinite(
      size.x
    ) ||
    !Number.isFinite(
      size.y
    ) ||
    !Number.isFinite(
      size.z
    )
  ) {

    throw new Error(
      "Invalid GLB bounding box"
    );
  }


  bboxStatus.textContent =
    "Valid";


  sizeStatus.textContent =
    `${size.x.toFixed(3)} × ${size.y.toFixed(3)} × ${size.z.toFixed(3)}`;


  console.log(
    `[${config.name}] Size`,
    size
  );


  console.log(
    `[${config.name}] Meshes`,
    meshCount
  );


  console.log(
    `[${config.name}] Materials`,
    materials.size
  );


  const maxDimension =
    Math.max(
      size.x,
      size.y,
      size.z
    );


  if (
    maxDimension <= 0 ||
    !Number.isFinite(
      maxDimension
    )
  ) {

    throw new Error(
      "Invalid GLB dimensions"
    );
  }


  loadedModel =
    gltf.scene;


  loadedModel.position.set(
    -center.x,
    -center.y,
    -center.z
  );


  loadedModel.scale.setScalar(
    1 /
    maxDimension
  );


  loadedModel.rotation.set(
    config.rotation?.x || 0,
    config.rotation?.y || 0,
    config.rotation?.z || 0
  );


  modelAnchor.add(
    loadedModel
  );


  const animations =
    gltf.animations || [];


  animationStatus.textContent =
    String(
      animations.length
    );


  const clipNames =
    animations.map(
      (
        clip,
        index
      ) => {

        return (
          clip.name ||
          `Clip ${index}`
        );
      }
    );


  clipStatus.textContent =
    clipNames.length > 0
      ? clipNames.join(", ")
      : "None";


  console.log(
    `[${config.name}] Animation Clips`,
    clipNames
  );


  selectedClipStatus.textContent =
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
        `animationIndex ${requestedIndex} is invalid. Available clips: ${clipNames.join(", ")}`
      );
    }


    const selectedClip =
      animations[
        requestedIndex
      ];


    selectedClipStatus.textContent =
      selectedClip.name ||
      `Clip ${requestedIndex}`;


    console.log(
      `[${config.name}] Selected Animation`,
      selectedClip.name ||
      `Clip ${requestedIndex}`
    );


    mixer =
      new THREE.AnimationMixer(
        loadedModel
      );


    activeAction =
      mixer.clipAction(
        selectedClip
      );


    activeAction.reset();


    activeAction.setLoop(
      THREE.LoopRepeat,
      Infinity
    );


    activeAction.play();
  }


  modelAnchor.visible =
    false;


  glbLoadStatus.textContent =
    "PASS";


  clearError();


  setStatus(
    `${config.name} ready`
  );


  updateControls();
}


/* =========================================================
   READY
========================================================= */

function systemReady() {

  return Boolean(
    poseLandmarker &&
    imageSegmenter &&
    renderer &&
    loadedModel &&
    activeModelConfig
  );
}


function updateControls() {

  const ready =
    systemReady();


  startButton.disabled =
    !ready ||
    cameraRunning;


  modelSelector.disabled =
    MODEL_REGISTRY.length === 0;


  if (
    cameraRunning
  ) {

    switchButton.disabled =
      false;

    captureButton.disabled =
      false;

    stopButton.disabled =
      false;

  } else {

    switchButton.disabled =
      true;

    captureButton.disabled =
      true;

    stopButton.disabled =
      true;
  }


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


    startButton.disabled =
      true;
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


    setStatus(
      `${activeModelConfig.name} running`
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


  if (
    modelAnchor
  ) {

    modelAnchor.visible =
      false;
  }


  clearOverlay();


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
   MODEL SWITCHING
========================================================= */

function changeModel(
  modelId
) {

  if (
    modelId ===
    activeModelId &&
    loadedModel
  ) {

    return;
  }


  const config =
    getModelConfig(
      modelId
    );


  if (
    !config
  ) {

    setError(
      new Error(
        `Model not found: ${modelId}`
      )
    );

    return;
  }


  if (
    cameraRunning
  ) {

    stopStream();
  }


  activeModelId =
    modelId;


  modelSelector.value =
    modelId;


  loadActiveModel();
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
   OBJECT-FIT COVER
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
   LANDMARK -> CANVAS
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


/* =========================================================
   CANVAS -> THREE
========================================================= */

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
   BODY TRACKING
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


    anchorStatus.textContent =
      "Low confidence";


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
  const leftShoulderThree =
  canvasToThree(
    ls
  );


const rightShoulderThree =
  canvasToThree(
    rs
  );


trackedBody.leftShoulderX =
  leftShoulderThree.x;


trackedBody.leftShoulderY =
  leftShoulderThree.y;


trackedBody.rightShoulderX =
  rightShoulderThree.x;


trackedBody.rightShoulderY =
  rightShoulderThree.y;

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
      result.confidenceMasks.length === 0
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
      maskCanvas.width !== width ||
      maskCanvas.height !== height
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


      rgba[targetIndex] =
        255;


      rgba[targetIndex + 1] =
        255;


      rgba[targetIndex + 2] =
        255;


      rgba[targetIndex + 3] =
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

  if (
    !activeModelConfig
  ) {

    return false;
  }


  /*
    ORBIT:
    occlude only on the BACK half.

    BESIDE:
    character stays outside torso,
    so human occlusion is not required.
  */

  if (
    activeModelConfig.behavior ===
    "ORBIT"
  ) {

    return (
      orbitDepth < 0
    );
  }


  return false;
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
  delta
) {

  if (
    !trackedBody.valid ||
    !activeModelConfig
  ) {

    modelAnchor.visible =
      false;


    return;
  }


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
    (
      trackedBody.shoulderWidth *
      0.65
      +
      trackedBody.torsoHeight *
      0.35
    );


  const baseScale =
    bodyReference *
    activeModelConfig.scaleMultiplier;


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


  modelAnchor.position.x =
    damp(
      modelAnchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.y =
    damp(
      modelAnchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.z =
    orbitDepth *
    0.1;


  const smoothScale =
    damp(
      modelAnchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  modelAnchor.scale.setScalar(
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


    modelAnchor.rotation.y =
      dampAngle(
        modelAnchor.rotation.y,
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


  modelAnchor.rotation.z =
    damp(
      modelAnchor.rotation.z,
      targetBank,
      ROTATION_SMOOTHING,
      delta
    );


  modelAnchor.visible =
    true;


  const depthLabel =
    orbitDepth >= 0
      ? "FRONT"
      : "BACK";


  const maskLabel =
    segmentationMaskReady
      ? "MASK STABLE"
      : "MASK WAIT";


  anchorStatus.textContent =
    `${activeModelConfig.name} | ORBIT ${depthLabel} | ${maskLabel} | scale ${smoothScale.toFixed(3)}`;
}


/* =========================================================
   BESIDE BEHAVIOR
========================================================= */

function updateBesideBehavior(
  delta
) {

  if (
    !trackedBody.valid ||
    !activeModelConfig
  ) {

    modelAnchor.visible =
      false;


    return;
  }


  /*
    Reset orbit depth because BESIDE
    does not use fake front/back orbit.
  */

  orbitDepth =
    0;


  const beside =
    activeModelConfig.beside || {};


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
    (
      trackedBody.shoulderWidth *
      0.65
      +
      trackedBody.torsoHeight *
      0.35
    );


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      activeModelConfig.scaleMultiplier,

      0.05,

      1.2
    );


  modelAnchor.position.x =
    damp(
      modelAnchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.y =
    damp(
      modelAnchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.z =
    0;


  const smoothScale =
    damp(
      modelAnchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  modelAnchor.scale.setScalar(
    smoothScale
  );


  /*
    Waveboy stays upright.
  */

  modelAnchor.rotation.y =
    dampAngle(
      modelAnchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  modelAnchor.rotation.z =
    damp(
      modelAnchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  modelAnchor.visible =
    true;


  const sideLabel =
    side > 0
      ? "RIGHT"
      : "LEFT";


  anchorStatus.textContent =
    `${activeModelConfig.name} | BESIDE ${sideLabel} | scale ${smoothScale.toFixed(3)}`;
}

/* =========================================================
   SHOULDER BEHAVIOR
========================================================= */

function updateShoulderBehavior(
  delta
) {

  if (
    !trackedBody.valid ||
    !activeModelConfig
  ) {

    modelAnchor.visible =
      false;

    return;
  }


  orbitDepth =
    0;


  const shoulder =
    activeModelConfig.shoulder || {};


  const side =
    shoulder.side === "left"
      ? "left"
      : "right";


  const offsetX =
    Number.isFinite(
      shoulder.offsetX
    )
      ? shoulder.offsetX
      : 0.3;


  const offsetY =
    Number.isFinite(
      shoulder.offsetY
    )
      ? shoulder.offsetY
      : 0.1;


  let targetX;
  let targetY;


  if (
    side === "left"
  ) {

    targetX =
      trackedBody.leftShoulderX -
      trackedBody.shoulderWidth *
      offsetX;


    targetY =
      trackedBody.leftShoulderY +
      trackedBody.torsoHeight *
      offsetY;

  } else {

    targetX =
      trackedBody.rightShoulderX +
      trackedBody.shoulderWidth *
      offsetX;


    targetY =
      trackedBody.rightShoulderY +
      trackedBody.torsoHeight *
      offsetY;
  }


  const bodyReference =
    (
      trackedBody.shoulderWidth *
      0.65
      +
      trackedBody.torsoHeight *
      0.35
    );


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      activeModelConfig.scaleMultiplier,

      0.05,

      1.2
    );


  modelAnchor.position.x =
    damp(
      modelAnchor.position.x,
      targetX,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.y =
    damp(
      modelAnchor.position.y,
      targetY,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.z =
    0;


  const smoothScale =
    damp(
      modelAnchor.scale.x,
      targetScale,
      SCALE_SMOOTHING,
      delta
    );


  modelAnchor.scale.setScalar(
    smoothScale
  );


  modelAnchor.rotation.y =
    dampAngle(
      modelAnchor.rotation.y,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  modelAnchor.rotation.z =
    damp(
      modelAnchor.rotation.z,
      0,
      ROTATION_SMOOTHING,
      delta
    );


  modelAnchor.visible =
    true;


  anchorStatus.textContent =
    `${activeModelConfig.name} | SHOULDER ${side.toUpperCase()} | scale ${smoothScale.toFixed(3)}`;
}

/* =========================================================
   BEHAVIOR ROUTER
========================================================= */

function updateModelBehavior(
  delta
) {

  if (
    !activeModelConfig
  ) {

    modelAnchor.visible =
      false;


    return;
  }


  switch (
    activeModelConfig.behavior
  ) {

    case "ORBIT":

      updateOrbitBehavior(
        delta
      );

      break;


    case "BESIDE":

      updateBesideBehavior(
        delta
      );
      break;

    case "SHOULDER":

      updateShoulderBehavior(
        delta
     );
    
      break;
    

    default:

      modelAnchor.visible =
        false;


      anchorStatus.textContent =
        `Unsupported behavior: ${activeModelConfig.behavior}`;
  }
}


/* =========================================================
   DEBUG POSE
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
    2 * dpr;


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
        "Unable to create capture image"
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
      `human-ar-${activeModelId}-${timestamp}.jpg`;


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


  if (
    mixer
  ) {

    mixer.update(
      delta
    );
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


        modelAnchor.visible =
          false;


        anchorStatus.textContent =
          "Hidden";
      }

    } catch (error) {

      setError(
        error
      );


      setStatus(
        "Pose detection error"
      );


      cameraRunning =
        false;


      return;
    }
  }


  updateModelBehavior(
    delta
  );


  renderer.render(
    scene,
    threeCamera
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


modelSelector.addEventListener(
  "change",
  event => {

    changeModel(
      event.target.value
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
   INIT
========================================================= */

console.log(
  "[Human AR] Milestone 8.3 initialized"
);


startButton.disabled =
  true;

switchButton.disabled =
  true;

captureButton.disabled =
  true;

stopButton.disabled =
  true;


initializeModelSelector();


try {

  initializeThree();

  loadActiveModel();

} catch (error) {

  setError(
    error
  );


  setStatus(
    "Three.js initialization failed"
  );
}


initializeMediaPipe();