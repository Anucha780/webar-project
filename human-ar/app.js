import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

import * as THREE
  from "https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js";

import { GLTFLoader }
  from "https://cdn.jsdelivr.net/npm/three@0.185.1/examples/jsm/loaders/GLTFLoader.js";

/* =========================================================
   DOM
========================================================= */

const video = document.querySelector("#camera");
const threeLayer = document.querySelector("#three-layer");
const overlay = document.querySelector("#pose-overlay");
const ctx = overlay.getContext("2d");

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
let facingMode = "user";

let poseLandmarker = null;

let cameraRunning = false;
let animationFrameId = null;
let lastVideoTime = -1;


/* =========================================================
   THREE.JS
========================================================= */

let scene = null;
let threeCamera = null;
let renderer = null;

let testSphere = null;

let loadedModel = null;

let mixer = null;

const clock =
  new THREE.Clock();


/* =========================================================
   GLB
========================================================= */

const MODEL_PATH =
  "./models/test-model.glb";


/* =========================================================
   LANDMARKS
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
   THREE INITIALIZATION
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
    Keep the primitive from M4.

    This stays as our known-good
    reference object.
  */

  const sphereGeometry =
    new THREE.SphereGeometry(
      0.04,
      24,
      16
    );

  const sphereMaterial =
    new THREE.MeshNormalMaterial();

  testSphere =
    new THREE.Mesh(
      sphereGeometry,
      sphereMaterial
    );

  testSphere.visible =
    false;

  scene.add(
    testSphere
  );


  /*
    Basic lights for GLB materials.
  */

  const ambient =
    new THREE.AmbientLight(
      0xffffff,
      1.8
    );

  scene.add(
    ambient
  );

  const directional =
    new THREE.DirectionalLight(
      0xffffff,
      2
    );

  directional.position.set(
    1,
    2,
    3
  );

  scene.add(
    directional
  );

  resizeThree();

  threeStatus.textContent =
    `Ready r${THREE.REVISION}`;
}


/* =========================================================
   GLB VALIDATION
========================================================= */

function loadAndValidateGLB() {

  glbPathStatus.textContent =
    MODEL_PATH;

  glbLoadStatus.textContent =
    "Loading...";

  const loader =
    new GLTFLoader();

  loader.load(

    MODEL_PATH,

    (gltf) => {

      console.log(
        "[GLB] Loaded",
        gltf
      );


      /* -----------------------------
         1. Scene
      ----------------------------- */

      if (!gltf.scene) {

        throw new Error(
          "GLB has no scene"
        );
      }


      /* -----------------------------
         2. Mesh inspection
      ----------------------------- */

      let meshCount = 0;
      let materialCount = 0;

      const materials =
        new Set();

      gltf.scene.traverse(
        (object) => {

          if (object.isMesh) {

            meshCount++;

            console.log(
              "[GLB Mesh]",
              object.name || "(unnamed)",
              object.geometry
            );

            if (object.material) {

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
          }
        }
      );

      materialCount =
        materials.size;

      meshStatus.textContent =
        String(meshCount);

      materialStatus.textContent =
        String(materialCount);


      /* -----------------------------
         3. Bounding Box
      ----------------------------- */

      const box =
        new THREE.Box3()
          .setFromObject(
            gltf.scene
          );

      if (box.isEmpty()) {

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
        "[GLB Bounding Box]",
        box
      );

      console.log(
        "[GLB Size]",
        size
      );

      console.log(
        "[GLB Center]",
        center
      );


      /* -----------------------------
         4. Center model
      ----------------------------- */

      gltf.scene.position.sub(
        center
      );


      /* -----------------------------
         5. Normalize scale

         Validation only.

         We scale the longest axis
         to approximately 0.18 scene units.
      ----------------------------- */

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

      const normalizedScale =
        0.18 /
        maxDimension;

      gltf.scene.scale.setScalar(
        normalizedScale
      );


      /* -----------------------------
         6. Animations
      ----------------------------- */

      const animations =
        gltf.animations || [];

      animationStatus.textContent =
        `${animations.length}`;

      if (
        animations.length > 0
      ) {

        const names =
          animations.map(
            (clip, index) =>
              clip.name ||
              `Clip ${index}`
          );

        clipStatus.textContent =
          names.join(", ");

        console.log(
          "[GLB Animations]",
          animations
        );

        /*
          IMPORTANT:

          We deliberately do NOT
          automatically choose an
          animation yet.

          M5 only validates clips.
        */

      } else {

        clipStatus.textContent =
          "None";
      }


      /* -----------------------------
         7. Add to scene

         Display near screen center
         for visual validation only.
      ----------------------------- */

      loadedModel =
        gltf.scene;

      loadedModel.position.set(
        0.5,
        0.5,
        0
      );

      loadedModel.visible =
        true;

      scene.add(
        loadedModel
      );

      glbLoadStatus.textContent =
        "PASS";

      setStatus(
        "GLB validated"
      );

      renderer.render(
        scene,
        threeCamera
      );

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

    (error) => {

      console.error(
        "[GLB Load Error]",
        error
      );

      glbLoadStatus.textContent =
        "FAILED";

      setError(
        new Error(
          "Unable to load GLB"
        )
      );

      setStatus(
        "GLB validation failed"
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

    if (
      glbLoadStatus.textContent ===
      "PASS"
    ) {

      setStatus(
        "M5 ready — start camera"
      );

    } else {

      setStatus(
        "Pose ready — validating GLB"
      );
    }

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
   CAMERA START
========================================================= */

async function startCamera() {

  if (
    !poseLandmarker ||
    !renderer
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
      "Camera + Pose running"
    );

    predictPose();

  } catch (error) {

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

  if (testSphere) {

    testSphere.visible =
      false;
  }

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
      rect.width * dpr
    );

  overlay.height =
    Math.round(
      rect.height * dpr
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

  threeCamera.left = 0;
  threeCamera.right = 1;
  threeCamera.top = 1;
  threeCamera.bottom = 0;

  threeCamera.updateProjectionMatrix();
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
   POSE → TEST SPHERE
========================================================= */

function updateAnchor(
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

  const canvasPoint =
    landmarkToCanvas(
      torso
    );

  const point =
    canvasToThree(
      canvasPoint
    );

  testSphere.visible =
    true;

  testSphere.position.set(
    point.x,
    point.y,
    0
  );

  anchorStatus.textContent =
    `TORSO ${point.x.toFixed(3)}, ${point.y.toFixed(3)}`;
}


/* =========================================================
   POSE LOOP
========================================================= */

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

    const result =
      poseLandmarker.detectForVideo(
        video,
        performance.now()
      );

    if (
      result.landmarks &&
      result.landmarks.length > 0
    ) {

      personStatus.textContent =
        "Detected";

      updateAnchor(
        result.landmarks[0]
      );

    } else {

      personStatus.textContent =
        "Not detected";

      testSphere.visible =
        false;
    }
  }


  /*
    M5 visual validation:

    Rotate GLB slowly in the screen center
    so we can inspect materials/geometry.
  */

  if (loadedModel) {

    loadedModel.rotation.y +=
      0.01;
  }


  if (mixer) {

    const delta =
      clock.getDelta();

    mixer.update(
      delta
    );
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
  "[Human AR] Milestone 5 initialized"
);

startButton.disabled =
  true;

try {

  initializeThree();

  loadAndValidateGLB();

} catch (error) {

  setError(
    error
  );

  setStatus(
    "Three.js initialization failed"
  );
}

initializePose();