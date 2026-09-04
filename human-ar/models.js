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

    enabled: true,

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

    enabled: true,

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

export function normalizeModelConfig(
  model
) {

  return {

    ...model,

    enabled:
      model.enabled !== false,

    scaleMultiplier:
      Number.isFinite(
        model.scaleMultiplier
      )
        ? model.scaleMultiplier
        : 1,

    rotation: {

      x:
        Number.isFinite(
          model.rotation?.x
        )
          ? model.rotation.x
          : 0,

      y:
        Number.isFinite(
          model.rotation?.y
        )
          ? model.rotation.y
          : 0,

      z:
        Number.isFinite(
          model.rotation?.z
        )
          ? model.rotation.z
          : 0
    },

    animationIndex:
      Number.isInteger(
        model.animationIndex
      )
        ? model.animationIndex
        : 0
  };
}

export const DEFAULT_MODEL_ID =
  "butterfly";


export function getModelConfig(
  modelId
) {

  const model =
    MODEL_REGISTRY.find(
      item =>
        item.id === modelId
    );


  if (
    !model
  ) {

    return null;
  }


  return normalizeModelConfig(
    model
  );
}