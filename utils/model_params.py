import numpy as np


model_dict = {

  "og_pos" : np.array([0, 0, 1]),

  "mp_size_l" : [0.2, 0.2, 0.2],

  "ai_pos" : np.array([[-3.3, -1.35,   3],
		               [ 3.3, -1.35,   3],
		               [ 3.3,  1.35,   3],
		               [-3.3,  1.35,   3],
		               [-2.6, -1.7, 3],
		               [ 2.6, -1.7, 3],
		               [ 2.6,  1.7, 3],
		               [-2.6,  1.7, 3]]),

  "ei_pos" : np.array([[-0.1, -0.1, -0.1],
		               [ 0.1, -0.1, -0.1],
		               [ 0.1,  0.1, -0.1],
		               [-0.1,  0.1, -0.1],
		               [-0.1, -0.1,  0.1],
		               [ 0.1, -0.1,  0.1],
		               [ 0.1,  0.1,  0.1],
		               [-0.1,  0.1,  0.1]]),
  

  "mp_file" : "model/cdprv3_mp.xml",
  "cable_file" : "model/cdprv3_cables.xml",
  "scene_file" : "model/scene.xml",
  "main_file" : "model/cdprv3.xml",
  "include_files" : ["cdprv3_mp.xml","cdprv3_cables.xml","scene.xml"],
  "tmin": 0.0,  # or something like 0.1 if you allow slack cables,
  "tmax": 5000.0  # max tension each cable can apply

}