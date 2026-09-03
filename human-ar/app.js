import {
  FilesetResolver,
  PoseLandmarker
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

let cameraRunning =
  false;

let animationFrameId =
  null;

let lastVideoTime =
  -1;

let lastFrameTimestamp =
  performance.now();


/* =========================================================
   THREE.JS
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


/*
  ขนาด character เทียบกับคน

  ใหญ่เกิน -> ลด
  เล็กเกิน -> เพิ่ม
*/

const MODEL_SCALE_MULTIPLIER =
  1.8;


/*
  Character จะอยู่ด้านไหนของคน

  "right" = ด้านขวาของหน้าจอ
  "left"  = ด้านซ้ายของหน้าจอ
*/

const CHARACTER_SIDE =
  "right";


/*
  ระยะห่างจากคน

  ค่านี้อิงกับ shoulder width
  ไม่ใช่ pixel ตายตัว
*/

const SIDE_DISTANCE =
  1.25;


/*
  Y Offset

  positive = ขึ้น
  negative = ลง
*/

const MODEL_OFFSET_Y =
  -0.03;


/*
  Rotation ของ GLB เอง

  ถ้าตัวละครหันผิดด้าน
  ค่อยจูนค่าตรงนี้ภายหลัง
*/

const MODEL_ROTATION_X =
  0;

const MODEL_ROTATION_Y =
  0;

const MODEL_ROTATION_Z =
  0;


/* =========================================================
   SMOOTHING
========================================================= */

const POSITION_SMOOTHING =
  10;

const SCALE_SMOOTHING =
  8;

const MIN_VISIBILITY =
  0.55;


/* =========================================================
   LANDMARK INDEX
========================================================= */

const LANDMARK = {

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
   SMOOTHING
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


/* =========================================================
   THREE
========================================================= */

function initializeThree() {

  try {

    threeStatus.textContent =
      "Initializing...";


    scene =
      new THREE.Scene();


    /*
      Screen-space coordinate system

      x = 0 → 1
      y = 0 → 1
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
   GLB LOAD + VALIDATE
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


        /* ---------------------------------
           Bounding box
        --------------------------------- */

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


        console.log(
          "[GLB Size]",
          size
        );


        console.log(
          "[GLB Center]",
          center
        );


        /* ---------------------------------
           Normalize
        --------------------------------- */

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


        /* ---------------------------------
           Inspect animations FIRST
        --------------------------------- */

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


        /*
          Important:

          เราไม่ guess ชื่อ animation

          หลังจาก inspect gltf.animations แล้ว
          ถ้ามี clip จริง จึงใช้ clip ตัวแรก
        */

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

        } else {

          console.log(
            "[GLB] No animations"
          );
        }


        modelAnchor.visible =
          false;


        glbLoadStatus.textContent =
          "PASS";


        errorStatus.textContent =
          "None";


        setStatus(
          "GLB ready — Character Side Mode"
        );


        renderer.render(
          scene,
          threeCamera
        );

      } catch (error) {

        glbLoadStatus.textContent =
          "FAILED";


        setError(
          error
        );


        setStatus(
          "GLB validation failed"
        );
      }
    },


    (progress) => {

      if (
        progress.total > 0
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
      "Ready";


    setStatus(
      "Ready — Start Camera"
    );


    startButton.disabled =
      false;

  } catch (error) {

    mediapipeStatus.textContent =
      "Failed";


    setError(
      error
    );


    setStatus(
      "MediaPipe failed"
    );
  }
}


/* =========================================================
   CAMERA
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


async function startCamera() {

  if (
    !poseLandmarker ||
    !renderer ||
    !loadedModel
  ) {

    setStatus(
      "System is not ready"
    );


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


    setStatus(
      "M6.3A Character Side Mode running"
    );


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


    startButton.disabled =
      false;
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


  placeholder.style.display =
    "flex";


  placeholder.textContent =
    "Camera stopped";


  switchButton.disabled =
    true;


  stopButton.disabled =
    true;


  startButton.disabled =
    !poseLandmarker ||
    !loadedModel;
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
   LANDMARK → CANVAS
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
   CANVAS → THREE
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

  if (!landmark) {
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


/* =========================================================
   CHARACTER BESIDE PERSON
========================================================= */

function updateCharacter(
  landmarks,
  delta
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


  if (!reliable) {

    modelAnchor.visible =
      false;


    anchorStatus.textContent =
      "Low confidence";


    return;
  }


  /* ---------------------------------
     Convert landmarks to display
  --------------------------------- */

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


  /* ---------------------------------
     Body centers
  --------------------------------- */

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


  const torsoCenter = {

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


  /* ---------------------------------
     Body dimensions
  --------------------------------- */

  const shoulderWidthPixels =
    distance2D(
      ls,
      rs
    );


  const torsoHeightPixels =
    distance2D(
      shoulderCenter,
      hipCenter
    );


  const shoulderWidth =
    shoulderWidthPixels /
    overlay.width;


  const torsoHeight =
    torsoHeightPixels /
    overlay.height;


  /* ---------------------------------
     Character side position
  --------------------------------- */

  const sideSign =
    CHARACTER_SIDE === "left"
      ? -1
      : 1;


  /*
    Character position in canvas pixels.

    การใช้ shoulderWidth ทำให้ระยะห่างจากคน
    ขยาย/ย่อตามขนาดคนด้วย
  */

  const characterCanvas = {

    x:
      torsoCenter.x +
      (
        sideSign *
        shoulderWidthPixels *
        SIDE_DISTANCE
      ),


    y:
      torsoCenter.y -
      (
        MODEL_OFFSET_Y *
        overlay.height
      )
  };


  const characterThree =
    canvasToThree(
      characterCanvas
    );


  /* ---------------------------------
     Scale
  --------------------------------- */

  const bodyReference =
    (
      shoulderWidth * 0.65 +
      torsoHeight * 0.35
    );


  const targetScale =
    THREE.MathUtils.clamp(

      bodyReference *
      MODEL_SCALE_MULTIPLIER,

      0.06,
      1.2
    );


  /* ---------------------------------
     Smooth movement
  --------------------------------- */

  modelAnchor.position.x =
    damp(
      modelAnchor.position.x,
      characterThree.x,
      POSITION_SMOOTHING,
      delta
    );


  modelAnchor.position.y =
    damp(
      modelAnchor.position.y,
      characterThree.y,
      POSITION_SMOOTHING,
      delta
    );


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
    Character ยืนตรงเอง
    ไม่เอียงตามไหล่ใน mode นี้
  */

  modelAnchor.rotation.z =
    0;


  modelAnchor.visible =
    true;


  anchorStatus.textContent =
    `${CHARACTER_SIDE.toUpperCase()} SIDE | x ${modelAnchor.position.x.toFixed(3)} | y ${modelAnchor.position.y.toFixed(3)} | scale ${smoothScale.toFixed(3)}`;
}


/* =========================================================
   DEBUG BODY
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


function drawPose(
  landmarks
) {

  clearOverlay();


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
   LOOP
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
      ) / 1000,
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
    video.currentTime !== lastVideoTime
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


        const landmarks =
          result.landmarks[0];


        drawPose(
          landmarks
        );


        updateCharacter(
          landmarks,
          delta
        );

      } else {

        personStatus.textContent =
          "Not detected";


        clearOverlay();


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


  renderer.render(
    scene,
    threeCamera
  );


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
  "[Human AR] Milestone 6.3A initialized"
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


  setStatus(
    "Three.js initialization failed"
  );
}


initializePose();