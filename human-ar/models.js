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

    behavior: "BESIDE",

    scaleMultiplier: 1.15,

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },

    animationIndex: 0,

    beside: {

      /*
        "right" = ด้านขวาบนหน้าจอ
      */

      side: "right",

      /*
        ระยะห่างจากกลางลำตัว
        คูณด้วย shoulder width
      */

      distance: 1.20,

      /*
        ปรับสูง/ต่ำ
        อิง torso height
      */

      offsetY: -0.10
    }
  }

];


export const DEFAULT_MODEL_ID =
  "butterfly";


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