/*
  ========================================================
  HUMAN AR — MODEL REGISTRY
  ========================================================

  Add every Human AR GLB model here.

  Current supported behavior:
  - ORBIT

  Future behaviors:
  - BESIDE
  - SHOULDER
  - STATIC
*/


export const MODEL_REGISTRY = [

  /*
    ======================================================
    MODEL 1 — BUTTERFLY
    ======================================================
  */

  {
    id: "butterfly",

    name: "Butterfly",

    path: "./models/test-model.glb",

    behavior: "ORBIT",

    scaleMultiplier: 1.35,

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },

    /*
      Animation index is validated only after
      gltf.animations has been inspected.

      0 = first real clip in the GLB.
    */

    animationIndex: 0
  },


  /*
    ======================================================
    MODEL 2 — WAVEBOY
    ======================================================
  */

  {
    id: "waveboy",

    name: "Waveboy",

    path: "./models/waveboy.glb",

    /*
      M8.2 tests ONLY model switching.

      Keep ORBIT temporarily so we don't mix
      model loading problems with a new behavior.
    */

    behavior: "ORBIT",

    /*
      Temporary starting value.

      Because app.js normalizes every GLB using its
      bounding box first, this is only the final
      body-relative size multiplier.

      We will tune it AFTER seeing the real model.
    */

    scaleMultiplier: 1.35,

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },

    /*
      The loader will first inspect:

      gltf.animations
      clip names
      number of clips

      and validate that this index exists.
    */

    animationIndex: 0
  }

];


/*
  Model loaded when Human AR starts.
*/

export const DEFAULT_MODEL_ID =
  "butterfly";


/*
  Find model configuration by ID.
*/

export function getModelConfig(
  modelId
) {

  return (
    MODEL_REGISTRY.find(
      model =>
        model.id === modelId
    ) || null
  );
}