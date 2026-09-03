/*
  ========================================================
  HUMAN AR — MODEL REGISTRY
  ========================================================

  ทุก GLB ที่จะนำเข้า Human AR
  ต้องประกาศที่นี่

  behavior ที่รองรับตอนนี้:
  - ORBIT

  Milestone ถัดไปจะเพิ่ม:
  - BESIDE
  - SHOULDER
  - STATIC
*/


export const MODEL_REGISTRY = [

  {
    id: "butterfly",

    name: "Butterfly",

    path: "./models/test-model.glb",

    behavior: "ORBIT",

    /*
      scaleMultiplier
      ใช้ปรับขนาดเฉพาะโมเดลตัวนี้
      โดยไม่กระทบโมเดลอื่น
    */

    scaleMultiplier: 1.35,


    /*
      rotation offset ของ GLB

      หน่วยเป็น radians
    */

    rotation: {
      x: 0,
      y: 0,
      z: 0
    },


    /*
      animationIndex

      เราไม่เดาชื่อ clip

      หลังโหลด GLB จะ inspect
      gltf.animations ก่อนเสมอ

      0 = clip แรกที่มีจริง
    */

    animationIndex: 0
  }

];


/*
  Default model
*/

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