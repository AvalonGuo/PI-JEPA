#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""\u52a0\u8f7d dr3d \u5bf9\u8c61/\u573a\u666f\uff0c\u6e32\u67d3\u7b2c\u4e00\u5e27 512 RGB -> PNG + meta json.\n\u7528\u6cd5: python dr3d/verify_render.py --obj can|bottle|cereal|mug|hammer|hammer+can ..."""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parent
DST = DR3D / 'distractors'
OBJ_XML = {'can': 'can/model.xml', 'bottle': 'bottle/model.xml',
           'cereal': 'cereal/model.xml', 'mug': 'mug/model.xml'}
HAMMER = ('/root/3DP/third_party/rrl-dependencies/mj_envs/mj_envs/'
          'hand_manipulation_suite/assets/DAPG_hammer.xml')


def load_model(obj, pos='0.22 -0.15 0.03', scale=1.0):
    import mujoco_py
    from mujoco_py import load_model_from_path
    tmp = None
    if obj == 'hammer':
        path = HAMMER
    elif obj in OBJ_XML:
        path = str(DST / OBJ_XML[obj])
    elif obj.startswith('hammer+'):
        name = obj.split('+', 1)[1]
        assert name in OBJ_XML, name
        sys.path.insert(0, str(DR3D))
        from test_load import compose_hammer_object
        tmp = compose_hammer_object(name, DST / OBJ_XML[name], pos, scale=scale)
        path = tmp
    else:
        raise SystemExit(f'unknown obj: {obj}')
    model = load_model_from_path(path)
    if tmp:
        os.remove(tmp)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--obj', default='can')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--camera', default=None)
    p.add_argument('--scale', type=float, default=1.0,
                   help='组合时网格缩放系数(默认 1.0)')
    p.add_argument('--out', default='/hy-tmp/3DP/dr3d_preview')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import mujoco_py
    model = load_model(a.obj, scale=a.scale)
    sim = mujoco_py.MjSim(model)
    sim.forward()

    cams = list(getattr(model, 'camera_names', []) or [])
    cam = a.camera or ('top' if 'top' in cams else (cams[0] if cams else None))
    if cam is not None and cam not in cams:
        cam = cams[0] if cams else None
    img = sim.render(a.size, a.size, camera_name=cam)
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    import imageio
    suf = f'_scale{a.scale:g}' if abs(a.scale - 1.0) > 1e-9 else ''
    png = os.path.join(a.out, f'{a.obj}{suf}_frame000.png')
    imageio.imwrite(png, img)
    meta = {'obj': a.obj, 'size': a.size, 'camera': cam, 'nbody': model.nbody,
            'ngeom': model.ngeom, 'nmesh': model.nmesh,
            'nq_nv': model.nq - model.nv, 'png': png}
    with open(os.path.join(a.out, f'{a.obj}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[render] {a.obj}: camera={cam} nbody={model.nbody} ngeom={model.ngeom} '
          f'nmesh={model.nmesh} -> {png}')


if __name__ == '__main__':
    main()
