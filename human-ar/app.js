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


/* =========================================================
   THREE.JS
========================================================= */

let scene =
  null;

let threeCamera =
  null;

let renderer =
  null;


/*
  The GLB itself will live inside modelAnchor.

  We move / scale modelAnchor according to
  MediaPipe body tracking.
*/

let modelAnchor =
  null;

let loadedModel =
  null;

let mixer =
  null;

let activeAction =
  null;


/*
  Base dimensions of the loaded GLB.
*/

let modelOriginalSize =
  null;


/*
  Animation timing.
*/

const clock =
  new THREE.Clock();


/* =========================================================
   GLB CONFIG
========================================================= */

const MODEL_PATH =
  "./models/test-model.glb";

/*
  We inspected this during M5.

  Do NOT guess another animation name.
*/

const ANIMATION_NAME =
  "Take 001";


/* =========================================================
   TRACKING CONFIG
========================================================= */

/*
  This number controls how large the model
  appears relative to shoulder width.

  It is NOT a world-unit scale.

  We intentionally keep this simple in M6.1.
*/

const MODEL_SCALE_MULTIPLIER =
  2.2;


/*
  Smoothing is intentionally minimal in M6.1.

  More advanced stabilization belongs to M6.2.
*/

const POSITION_LERP =
  0.35;

const SCALE_LERP =
  0.25;


/* =========================================================
   MEDIAPIPE LANDMARKS
========================================================= */

const LANDMARK = {

  NOSE: 0,

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_HIP: 23,
  RIGHT_HIP: 24
};


/* =========================================================
   DEBUG HELPERS
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
      Orthographic camera.

      This means our scene is still
      SCREEN-SPACE AR.

      x:
      0 = left
      1 = right

      y:
      0 = bottom
      1 = top
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


    /*
      GLB body anchor.

      The model sits inside this group.
      We move this group based on MediaPipe.
    */

    modelAnchor =
      new THREE.Group();

    modelAnchor.visible =
      false;

    modelAnchor.position.set(
      0.5,
      0.5,
      0
    );

    scene.add(
      modelAnchor
    );


    /*
      Lighting for GLB materials.
    */

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


    console.log(
      `[Human AR] Three.js revision ${THREE.REVISION}`
    );

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
   GLB LOAD + VALIDATION
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

        console.log(
          "[GLB] Loaded",
          gltf
        );


        /* ----------------------------------
           Scene check
        ---------------------------------- */

        if (!gltf.scene) {

          throw new Error(
            "GLB has no scene"
          );
        }


        /* ----------------------------------
           Mesh / material check
        ---------------------------------- */

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


            /*
              Make sure shadows/material updates
              won't introduce stale transforms.
            */

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


        /* ----------------------------------
           Bounding box
        ---------------------------------- */

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


        modelOriginalSize =
          size.clone();


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


        /* ----------------------------------
           Center GLB locally
        ---------------------------------- */

        loadedModel =
          gltf.scene;


        /*
          Move the MODEL relative to its anchor.

          modelAnchor itself remains at (0,0,0)
          locally.

          This is important because later
          MediaPipe moves modelAnchor.
        */

        loadedModel.position.set(
          -center.x,
          -center.y,
          -center.z
        );


        /*
          Normalize original GLB size.

          Its largest dimension becomes 1.

          From this point onward, body tracking
          controls modelAnchor.scale.
        */

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


        modelAnchor.add(
          loadedModel
        );


        /* ----------------------------------
           Animation inspection
        ---------------------------------- */

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


        /* ----------------------------------
           Exact animation selection
        ---------------------------------- */

        const selectedClip =
          THREE.AnimationClip.findByName(
            animations,
            ANIMATION_NAME
          );


        if (
          animations.length > 0 &&
          !selectedClip
        ) {

          throw new Error(
            `Animation "${ANIMATION_NAME}" not found`
          );
        }


        if (
          selectedClip
        ) {

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


          console.log(
            `[GLB] Playing animation: ${selectedClip.name}`
          );
        }


        /*
          Do not show model until a person
          is detected.
        */

        modelAnchor.visible =
          false;


        glbLoadStatus.textContent =
          "PASS";


        setStatus(
          `GLB ready — animation: ${ANIMATION_NAME}`
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
        "GLB load failed"
      );
    }
  );
}


/* =========================================================
   MEDIAPIPE INITIALIZATION
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
      "Pose + GLB ready — start camera"
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


    startButton.disabled =
      true;
  }
}


/* =========================================================
   CAMERA DISPLAY
========================================================= */

function updateCameraDisplay() {

  /*
    Front camera preview = mirror.

    Raw video sent into MediaPipe
    remains unchanged.
  */

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


    clock.start();


    setStatus(
      "Camera + Pose + GLB running"
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


/* =========================================================
   SWITCH CAMERA
========================================================= */

async function switchCamera() {

  facingMode =
    facingMode === "user"
      ? "environment"
      : "user";


  stopStream();


  await startCamera();
}


/* =========================================================
   SIZE
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
   OBJECT-FIT COVER MAPPING
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


  /*
    Match mirrored front-camera preview.
  */

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
   CANVAS → THREE SCREEN COORDINATES
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
   DISTANCE
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


/* =========================================================
   GLB BODY ANCHOR
========================================================= */

function updateModelAnchor(
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
    Convert torso to visible canvas location.
  */

  const torsoCanvas =
    landmarkToCanvas(
      torso
    );


  const torsoThree =
    canvasToThree(
      torsoCanvas
    );


  /*
    Shoulder points use exactly the same
    camera crop / mirror correction.
  */

  const leftCanvas =
    landmarkToCanvas(
      leftShoulder
    );


  const rightCanvas =
    landmarkToCanvas(
      rightShoulder
    );


  /*
    Convert shoulder width to normalized
    screen units.

    Larger person in frame
    → larger shoulder distance
    → larger GLB.
  */

  const shoulderDistancePixels =
    distance2D(
      leftCanvas,
      rightCanvas
    );


  const shoulderScreenWidth =
    shoulderDistancePixels /
    overlay.width;


  const targetScale =
    shoulderScreenWidth *
    MODEL_SCALE_MULTIPLIER;


  /*
    Basic safety clamp.

    Prevent wild scale when landmarks
    briefly become unstable.
  */

  const safeScale =
    THREE.MathUtils.clamp(
      targetScale,
      0.08,
      1.2
    );


  /*
    Position smoothing.
  */

  modelAnchor.position.x =
    THREE.MathUtils.lerp(
      modelAnchor.position.x,
      torsoThree.x,
      POSITION_LERP
    );


  modelAnchor.position.y =
    THREE.MathUtils.lerp(
      modelAnchor.position.y,
      torsoThree.y,
      POSITION_LERP
    );


  modelAnchor.position.z =
    0;


  /*
    Uniform scale smoothing.
  */

  const currentScale =
    modelAnchor.scale.x;


  const smoothScale =
    THREE.MathUtils.lerp(
      currentScale,
      safeScale,
      SCALE_LERP
    );


  modelAnchor.scale.setScalar(
    smoothScale
  );


  modelAnchor.visible =
    true;


  anchorStatus.textContent =
    `TORSO ${torsoThree.x.toFixed(3)}, ${torsoThree.y.toFixed(3)} | scale ${smoothScale.toFixed(3)}`;
}


/* =========================================================
   DEBUG POSE DRAWING
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
    5 *
    dpr;


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
    `${10 * dpr}px Arial`;


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
    2 *
    dpr;


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


  /*
    Update GLB animation independently
    from pose detection frequency.
  */

  const delta =
    clock.getDelta();


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


        updateModelAnchor(
          landmarks
        );

      } else {

        personStatus.textContent =
          "Not detected";


        clearOverlay();


        if (
          modelAnchor
        ) {

          modelAnchor.visible =
            false;
        }


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
   INITIALIZATION
========================================================= */

console.log(
  "[Human AR] Milestone 6.1 initialized"
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