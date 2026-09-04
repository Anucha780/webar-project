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

      /*
        screen-right / screen-left

        รอบนี้เราใช้ตำแหน่งบนหน้าจอจริง
        จึงไม่สับสนระหว่างกล้องหน้าและกล้องหลัง
      */

      side: "right",

      /*
        ระยะออกจากไหล่
      */

      offsetX: 0.55,

      /*
        ยกตัวขึ้นเหนือแนวไหล่เล็กน้อย
      */

      offsetY: 0.25
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