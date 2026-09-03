import {
  FilesetResolver,
  PoseLandmarker,
  ImageSegmenter
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

import * as THREE from "three";

import {
  GLTFLoader
} from "three/addons/loaders/GLTFLoader.js";


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

const mediapipeStatus =
  document.querySelector("#mediapipe-status");

const threeStatus =
  document.querySelector("#three-status");

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

const cameraStatus =
  document.querySelector("#camera-status");

const personStatus =
  document.querySelector("#person-status");

const anchorStatus =
  document.querySelector("#anchor-status");

const errorStatus =
  document.querySelector("#error-status");


/* =========================================================
   STATE
========================================================= */

let stream = null;

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
   SEGMENTATION STATE
========================================================= */

let segmentationBusy =
  false;

let lastSegmentationTime =
  0;

let segmentationMaskReady =
  false;


/*
  เพิ่มจาก 12 FPS → 20 FPS

  ถ้ามือถือเครื่องไหนหนักมาก
  ค่อยลดกลับเป็น 15
*/

const SEGMENTATION_INTERVAL_MS =
  1000 / 20;


/*
  Confidence threshold
*/

const MASK_CONFIDENCE_LOW =
  0.12;

const MASK_CONFIDENCE_HIGH =
  0.52;


/*
  ลดจาก 1.04 → 1.02
  ลด halo รอบตัวคน
*/

const MASK_EXPANSION =
  1.02;


/*
  Temporal smoothing

  ค่ายิ่งสูง = mask ใหม่มีผลเร็ว
  ค่ายิ่งต่ำ = เนียน แต่ตาม movement ช้ากว่า

  0.70 เหมาะกับกล้องมือถือทั่วไป
*/

const MASK_TEMPORAL_ALPHA =
  0.70;


/*
  เราจะเก็บ mask ก่อนหน้าไว้
  แล้วผสมกับ mask ใหม่
*/

let previousMaskValues =
  null;


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

let loadedModel =
  null;

let mixer =
  null;

let activeAction =
  null;


/* =========================================================
   MODEL
========================================================= */

const MODEL_PATH =
  "./models/test-model.glb";


const MODEL_SCALE_MULTIPLIER =
  1.35;


const MODEL_ROTATION_X =
  0;

const MODEL_ROTATION_Y =
  0;

const MODEL_ROTATION_Z =
  0;


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
   LANDMARK
========================================================= */

const LANDMARK = {

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_HIP: 23,
  RIGHT_HIP: 24
};


/* =========================================================
   ORBIT STATE
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
    0.2
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
      -smoothing * delta
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
    difference > Math.PI
  ) {

    difference -=
      Math.PI * 2;
  }


  while (
    difference < -Math.PI
  ) {

    difference +=
      Math.PI * 2;
  }


  const factor =
    1 -
    Math.exp(
      -smoothing * delta
    );


  return (
    current +
    difference *
    factor
  );
}


/* =========================================================
   THREE
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
   GLB LOAD
========================================================= */

function loadGLB() {

  glbPathStatus.textContent =
    MODEL_PATH;


  glbLoadStatus.textContent =
    "Loading...";


  const loader =
    new GLTFLoader();


  loader.load(

    MODEL_PATH,


    (gltf) => {

      try {

        if (!gltf.scene) {

          throw new Error(
            "GLB has no scene"
          );
        }


        let meshCount =
          0;


        const materials =
          new Set();


        gltf.scene.traverse(
          (object) => {

            if (!object.isMesh) {
              return;
            }


            meshCount++;


            object.frustumCulled =
              false;


            if (!object.material) {
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


        bboxStatus.textContent =
          "Valid";


        sizeStatus.textContent =
          `${size.x.toFixed(3)} × ${size.y.toFixed(3)} × ${size.z.toFixed(3)}`;


        loadedModel =
          gltf.scene;


        loadedModel.position.set(
          -center.x,
          -center.y,
          -center.z
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


        loadedModel.scale.setScalar(
          1 / maxDimension
        );


        loadedModel.rotation.set(
          MODEL_ROTATION_X,
          MODEL_ROTATION_Y,
          MODEL_ROTATION_Z
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
            (clip, index) =>
              clip.name ||
              `Clip ${index}`
          );


        clipStatus.textContent =
          clipNames.length > 0
            ? clipNames.join(", ")
            : "None";


        console.log(
          "[GLB Animation Clips]",
          clipNames
        );


        if (
          animations.length > 0
        ) {

          const selectedClip =
            animations[0];


          console.log(
            "[GLB Selected Animation]",
            selectedClip.name
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


        errorStatus.textContent =
          "None";


        updateStartButton();

      } catch (error) {

        glbLoadStatus.textContent =
          "FAILED";


        setError(
          error
        );
      }
    },


    undefined,


    () => {

      glbLoadStatus.textContent =
        "FAILED";


      setError(
        new Error(
          "Unable to load GLB"
        )
      );
    }
  );
}


/* =========================================================
   READY
========================================================= */

function updateStartButton() {

  const ready =
    Boolean(
      poseLandmarker &&
      imageSegmenter &&
      renderer &&
      loadedModel
    );


  startButton.disabled =
    !ready;


  if (ready) {

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


    errorStatus.textContent =
      "None";


    updateStartButton();

  } catch (error) {

    mediapipeStatus.textContent =
      "Failed";


    setError(
      error
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
    !poseLandmarker ||
    !imageSegmenter ||
    !renderer ||
    !loadedModel
  ) {

    return;
  }


  errorStatus.textContent =
    "None";


  startButton.disabled =
    true;


  try {

    if (stream) {

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


    resizeOverlay();

    resizeThree();


    switchButton.disabled =
      false;


    stopButton.disabled =
      false;


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
      "M6.4A Segmentation Stabilization running"
    );


    predictPose();

  } catch (error) {

    setError(
      error
    );


    updateStartButton();
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


  if (stream) {

    stream
      .getTracks()
      .forEach(
        track =>
          track.stop()
      );
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


  modelAnchor.visible =
    false;


  clearOverlay();


  cameraStatus.textContent =
    "Stopped";


  personStatus.textContent =
    "Not detected";


  anchorStatus.textContent =
    "Hidden";


  placeholder.style.display =
    "flex";


  placeholder.textContent =
    "Camera stopped";


  switchButton.disabled =
    true;


  stopButton.disabled =
    true;


  updateStartButton();
}


function stopCamera() {

  stopStream();
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


  compositeCanvas.width =
    overlay.width;


  compositeCanvas.height =
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
    !sourceHeight
  ) {

    return null;
  }


  const scale =
    Math.max(
      displayWidth / sourceWidth,
      displayHeight / sourceHeight
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
   LANDMARK MAPPING
========================================================= */

function landmarkToCanvas(
  landmark
) {

  const transform =
    getCoverTransform();


  let sourceX =
    landmark.x *
    transform.sourceWidth;


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
      (
        transform.renderedWidth /
        transform.sourceWidth
      ) -
      transform.cropX,


    y:
      landmark.y *
      transform.sourceHeight *
      (
        transform.renderedHeight /
        transform.sourceHeight
      ) -
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

  if (!landmark) {
    return false;
  }


  return (
    landmark.visibility ===
      undefined ||
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


  if (
    !landmarkReliable(ls) ||
    !landmarkReliable(rs) ||
    !landmarkReliable(lh) ||
    !landmarkReliable(rh)
  ) {

    trackedBody.valid =
      false;


    return;
  }


  const pLS =
    landmarkToCanvas(ls);

  const pRS =
    landmarkToCanvas(rs);

  const pLH =
    landmarkToCanvas(lh);

  const pRH =
    landmarkToCanvas(rh);


  const shoulderCenter = {

    x:
      (
        pLS.x +
        pRS.x
      ) / 2,

    y:
      (
        pLS.y +
        pRS.y
      ) / 2
  };


  const hipCenter = {

    x:
      (
        pLH.x +
        pRH.x
      ) / 2,

    y:
      (
        pLH.y +
        pRH.y
      ) / 2
  };


  const center =
    canvasToThree({

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
    });


  trackedBody.centerX =
    center.x;


  trackedBody.centerY =
    center.y;


  trackedBody.shoulderWidth =
    distance2D(
      pLS,
      pRS
    ) /
    overlay.width;


  trackedBody.torsoHeight =
    distance2D(
      shoulderCenter,
      hipCenter
    ) /
    overlay.height;


  trackedBody.valid =
    true;
}


/* =========================================================
   SEGMENTATION REQUEST
========================================================= */

function requestSegmentation(
  timestamp
) {

  if (
    segmentationBusy ||
    !trackedBody.valid ||
    !imageSegmenter ||
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


/* =========================================================
   TEMPORALLY SMOOTHED SEGMENTATION
========================================================= */

function handleSegmentationResult(
  result
) {

  try {

    if (
      !result?.confidenceMasks?.length
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


    let j =
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


      rgba[j++] =
        255;

      rgba[j++] =
        255;

      rgba[j++] =
        255;

      rgba[j++] =
        alpha;
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
   COVER DRAW
========================================================= */

function drawCoverSource(
  targetContext,
  source,
  transform,
  expansion = 1
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
    facingMode === "user"
  ) {

    targetContext.translate(
      overlay.width,
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
   FULL HUMAN OCCLUSION
========================================================= */

function drawFullHumanOcclusion() {

  if (
    !segmentationMaskReady ||
    orbitDepth >= 0
  ) {

    return;
  }


  const transform =
    getCoverTransform();


  if (!transform) {
    return;
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
    1
  );


  compositeCtx.globalCompositeOperation =
    "destination-in";


  drawCoverSource(
    compositeCtx,
    maskCanvas,
    transform,
    MASK_EXPANSION
  );


  compositeCtx.globalCompositeOperation =
    "source-over";


  ctx.drawImage(
    compositeCanvas,
    0,
    0
  );
}


/* =========================================================
   ORBIT
========================================================= */

function updateOrbit(
  delta
) {

  if (
    !trackedBody.valid
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


  const targetX =
    trackedBody.centerX +
    orbitX *
    radiusX;


  const targetY =
    trackedBody.centerY +
    trackedBody.torsoHeight *
    ORBIT_CENTER_Y +
    orbitY *
    radiusY;


  const bodyReference =
    (
      trackedBody.shoulderWidth *
      0.65 +

      trackedBody.torsoHeight *
      0.35
    );


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      MODEL_SCALE_MULTIPLIER *
      (
        1 +
        orbitDepth *
        ORBIT_DEPTH_SCALE
      ),

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
      -orbitX * 12
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
    `ORBIT ${depthLabel} | ${maskLabel} | scale ${smoothScale.toFixed(3)}`;
}


/* =========================================================
   DEBUG POSE
========================================================= */

function drawPoseDebug(
  landmarks
) {

  const ids = [
    LANDMARK.LEFT_SHOULDER,
    LANDMARK.RIGHT_SHOULDER,
    LANDMARK.RIGHT_HIP,
    LANDMARK.LEFT_HIP
  ];


  const points =
    ids.map(
      index =>
        landmarkToCanvas(
          landmarks[index]
        )
    );


  const dpr =
    window.devicePixelRatio || 1;


  ctx.beginPath();


  ctx.moveTo(
    points[0].x,
    points[0].y
  );


  for (
    let i = 1;
    i < points.length;
    i++
  ) {

    ctx.lineTo(
      points[i].x,
      points[i].y
    );
  }


  ctx.closePath();


  ctx.strokeStyle =
    "#00ff88";


  ctx.lineWidth =
    2 * dpr;


  ctx.stroke();
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


  if (mixer) {

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
        result.landmarks?.length
      ) {

        latestLandmarks =
          result.landmarks[0];


        personStatus.textContent =
          "Detected";


        updateTrackedBody(
          latestLandmarks
        );


        requestSegmentation(
          now
        );

      } else {

        latestLandmarks =
          null;


        trackedBody.valid =
          false;


        segmentationMaskReady =
          false;


        previousMaskValues =
          null;


        modelAnchor.visible =
          false;


        personStatus.textContent =
          "Not detected";
      }

    } catch (error) {

      setError(
        error
      );
    }
  }


  updateOrbit(
    delta
  );


  renderer.render(
    scene,
    threeCamera
  );


  clearOverlay();


  if (
    latestLandmarks &&
    trackedBody.valid
  ) {

    drawFullHumanOcclusion();


    drawPoseDebug(
      latestLandmarks
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
  "[Human AR] Milestone 6.4A initialized"
);


startButton.disabled =
  true;


try {

  initializeThree();

  loadGLB();

} catch (error) {

  setError(
    error
  );
}


initializeMediaPipe();