use phf::phf_map;

// mesh annotations came from mediapipe python repo
static MESH_ANNOTATIONS: phf::Map<&'static str, Vec<usize> = phf_map! {
  "silhouette" => vec![
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ],

  "lipsUpperOuter" => vec![61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  "lipsLowerOuter" => vec![146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  "lipsUpperInner" => vec![78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  "lipsLowerInner" => vec![78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  "rightEyeUpper0" => vec![246, 161, 160, 159, 158, 157, 173],
  "rightEyeLower0" => vec![33, 7, 163, 144, 145, 153, 154, 155, 133],
  "rightEyeUpper1" => vec![247, 30, 29, 27, 28, 56, 190],
  "rightEyeLower1" => vec![130, 25, 110, 24, 23, 22, 26, 112, 243],
  "rightEyeUpper2" => vec![113, 225, 224, 223, 222, 221, 189],
  "rightEyeLower2" => vec![226, 31, 228, 229, 230, 231, 232, 233, 244],
  "rightEyeLower3" => vec![143, 111, 117, 118, 119, 120, 121, 128, 245],

  "rightEyebrowUpper" => vec![156, 70, 63, 105, 66, 107, 55, 193],
  "rightEyebrowLower" => vec![35, 124, 46, 53, 52, 65],

  "rightEyeIris" => vec![473, 474, 475, 476, 477],

  "leftEyeUpper0" => vec![466, 388, 387, 386, 385, 384, 398],
  "leftEyeLower0" => vec![263, 249, 390, 373, 374, 380, 381, 382, 362],
  "leftEyeUpper1" => vec![467, 260, 259, 257, 258, 286, 414],
  "leftEyeLower1" => vec![359, 255, 339, 254, 253, 252, 256, 341, 463],
  "leftEyeUpper2" => vec![342, 445, 444, 443, 442, 441, 413],
  "leftEyeLower2" => vec![446, 261, 448, 449, 450, 451, 452, 453, 464],
  "leftEyeLower3" => vec![372, 340, 346, 347, 348, 349, 350, 357, 465],

  "leftEyebrowUpper" => vec![383, 300, 293, 334, 296, 336, 285, 417],
  "leftEyebrowLower" => vec![265, 353, 276, 283, 282, 295],

  "leftEyeIris" => vec![468, 469, 470, 471, 472],

  "midwayBetweenEyes" => vec![168],

  "noseTip" => vec![1],
  "noseBottom" => vec![2],
  "noseRightCorner" => vec![98],
  "noseLeftCorner" => vec![327],

  "rightCheek" => vec![205],
  "leftCheek" => vec![425],
};
