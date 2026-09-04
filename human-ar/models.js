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
  },


  /*
    Temporary model สำหรับทดสอบ HEAD_ATTACH

    ใช้ waveboy.glb ซ้ำก่อน
    หลังจาก HEAD_ATTACH ผ่านแล้วค่อยเอาออก
    หรือเปลี่ยนเป็น asset จริง
  */

  {
    id: "headtest",

    name: "Head Test GLB",

    path: "./models/waveboy.glb",

    behavior: "HEAD_ATTACH",

    scaleMultiplier: 0.55,

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },

    animationIndex: 0,

    head: {

      /*
        0 = อยู่กึ่งกลางหัว
      */

      offsetX: 0,

      /*
        ค่าบวก = ขยับขึ้นเหนือหัว
      */

      offsetY: 0.6,

      /*
        scale เพิ่มเติมเฉพาะ HEAD_ATTACH
      */

      scaleFactor: 1
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