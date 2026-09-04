/*
  ========================================================
  HUMAN AR — MODEL REGISTRY
  ========================================================
*/

export const MODEL_REGISTRY = [

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

    animationIndex: 0
  },


  {
    id: "waveboy",

    name: "Waveboy",

    path: "./models/waveboy.glb",

    behavior: "SHOULDER",

    scaleMultiplier: 1.05,

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },

    animationIndex: 0,

    shoulder: {
      side: "right",
      offsetX: 0.55,
      offsetY: 0.25
    }
  }

];


export const DEFAULT_MODEL_ID =
  "butterfly";


export function getModelConfig(
  modelId
) {

  return MODEL_REGISTRY.find(
    model =>
      model.id === modelId
  ) || null;
}