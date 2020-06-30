import numpy as np
import json

objp = np.array([
    [5.9363, 2.9128, 0],
    [4, 2.4120, 0],
    [-4, 2.4120, 0],
    [-5.9363, 2.9128, 0],
    [-7.3134, -2.4120, 0],
    [-5.3771, -2.9128, 0],
    [5.3771, -2.9128, 0],
    [7.3134, -2.4120, 0]
], dtype=np.float32)

TargetModel = open('data/misc/TargetModel.mdl', 'w+')

TargetModel.write(json.dumps(objp.tolist()))

TargetModel.close()