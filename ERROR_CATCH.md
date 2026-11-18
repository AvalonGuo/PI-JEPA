# Error Catching
- `Aborted (core dumped)` right after the ckpt print of a periodic in-loop rollout
  (e.g. epoch 400 with `--rollout-every 200`): `run_rollout` used to build a fresh
  MuJoCo env per rollout and never release the previous one. It now closes the env,
  drops the runner and runs `gc` + `torch.cuda.empty_cache()` in a `finally`
  (`PI-JEPA/pijepa_common.py`), and logs rss/cgroup/vram before and after
  every rollout so an accumulation shows up before it aborts.
- RuntimeError: Fail to initialize OpenGL: enter the following command in the terminal
```bash
unset LD_PRELOAD
```

- AttributeError: module 'distutils' has no attribute 'version': `pip install setuptools==59.5.0`

- mujoco rendering is not using gpu: check this [link](https://github.com/openai/mujoco-py/issues/493)

- `AssertionError: Default values can only be a CfgNode or {<class 'int'>, <class 'bool'>, <class 'str'>, <class 'float'>, <class 'list'>, <class 'tuple'>}`: make sure yacs version, and `pip install yacs==0.1.8`

- `AttributeError: Can't get attribute '_make_function' on <module 'cloudpickle.cloudpickle' `: make sure cloudpickle version, and `pip install cloudpickle --upgrade`

- `*** SystemError: initialization of _internal failed without raising an exception`:
```bash
pip uninstall numba
pip install -U numba
```

- `File "/home/yanjieze/miniconda3/envs/dex/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 1649, in verify_ninja_availability
    raise RuntimeError("Ninja is required to load C++ extensions")
RuntimeError: Ninja is required to load C++ extensions`
```bash
sudo apt-get install ninja-build
```

- error when compiling mujoco-py 
```
Error compiling Cython file:
------------------------------------------------------------
...
    See c_warning_callback, which is the C wrapper to the user defined function
    '''
    global py_warning_callback
    global mju_user_warning
    py_warning_callback = warn
    mju_user_warning = c_warning_callback
                       ^
------------------------------------------------------------
```
solution
```bash
pip install Cython==0.29.35
```

- error when compiling mujoco-py f
```
2.1.2.14/mujoco_py/gl/eglshim.c:4:10: fatal error: GL/glew.h: No such file or directory
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
compilation terminated.
```
solution
```bash
sudo apt-get install libglew-dev
```

- *** RuntimeError: CUDA error: no kernel image is available for execution on the device
this occurs when using the functions from pytorch3d. reinstalling pytorch3d would solve the problem.
```
pip uninstall pytorch3d
cd third_party/pytorch3d
pip install -e .
cd ../..
```


- ImportError: cannot import name '_C' from 'pytorch3d' (/home/yanjieze/projects/diffusion-policy-for-dex/third_party/pytorch3d/pytorch3d/__init__.py)
```
pip uninstall pytorch3d
cd third_party/pytorch3d
pip install -e .
cd ../..
```

- wandb video error:
```
X Error of failed request:  BadAccess (attempt to access private resource denied)
  Major opcode of failed request:  152 (GLX)
  Minor opcode of failed request:  5 (X_GLXMakeCurrent)
  Serial number of failed request:  112
  Current serial number in output stream:  112
```
solution:
```bash
export QT_GRAPHICSSYSTEM=native
```


- ImportError("/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/gu/anaconda3/lib/python3.9/site-packages/pandas/_libs/window/aggregations.cpython-39-x86_64-linux-gnu.so)")
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install gcc-4.9
sudo apt-get upgrade libstdc++6
```
check by:
```
strings /usr/lib/arm-linux-gnueabihf/libstdc++.so.6 | grep GLIBCXX
```

- installing the third_party/gym-0.21.0 package when running with pip==24, leading to the following error:
```
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```
To solve this issue, run pip install pip==21 in addition to the setuptools downgrade already provided. See [Issue](https://github.com/YanjieZe/3D-Diffusion-Policy/issues/20)


- Error:

        ImportError: cannot import name 'cached_download' from 'huggingface_hub' (~/mambaforge/envs/dp3/lib/python3.8/site-packages/huggingface_hub/__init__.py)
  
This is because only huggingface_hub lower than 0.26 supports cached_download, try lower version can be solved:

    pip install huggingface_hub==0.25.2 

- Error:

        ERROR: Requested gym==0.21.0 from file:///~/3D-Diffusion-Policy/third_party/gym-0.21.0 has invalid metadata: Expected end or semicolon (after version specifier)
            opencv-python>=3.
                         ~~~^

This is because no specific opencv-python version, go to 3D-Diffusion-Policy/third_party/gym-0.21.0/setup.py and line 20, add 0 at the end of opencv-python>=3. just like this [issue](https://github.com/YanjieZe/3D-Diffusion-Policy/issues/92)

- wandb video error <TypeError: must be real number, not NoneType> in moviepy ffmpeg_writer
```
  File ".../moviepy/video/io/ffmpeg_writer.py", line 88, in __init__
    '-r', '%.02f' % fps,
TypeError: must be real number, not NoneType
```
Triggered by any `wandb.Video(numpy_frames, fps=..., format="mp4")` (adroit_runner.py:413 / metaworld_runner.py:431 / dexart_runner.py:160).
Cause: `moviepy 1.0.3` + `decorator 5.x`. moviepy's `use_clip_fps_by_default` (moviepy/decorators.py:107-138)
locates the `fps` argument by **position** via `f.__code__.co_varnames[1:]`; with decorator>=5 the generated
wrappers forward `*args/**kw`, so `fps` never appears as an argument and stays `None` -> `'%.02f' % None`.
moviepy 1.0.3 declares `Requires-Dist: decorator <5.0,>=4.0.2` (moviepy-1.0.3.dist-info/METADATA:26), and
the uv install line in QUICK_START.md/INSTALL.md pulled decorator 5.x.
Solution:
```bash
uv pip install --python .venv/bin/python "decorator==4.4.2"      # venv has no pip module, use uv
/root/3DP/.venv/bin/python -c "import numpy as np, wandb; wandb.Video(np.zeros((4,84,84,3),dtype='uint8'), fps=10, format='mp4')"
```
Only `ipython`/`ipdb` depend on decorator (no >=5 requirement), so the downgrade is safe.
(`format='gif'` goes through imageio and is unaffected; handing an existing mp4 path to
`wandb.Video(path, format="mp4")` also works - useful as an env-independent workaround.)

- mujoco_py: `model.camera_name2id(name)` **raises** ValueError for a missing camera
```
ValueError: No "camera" with name top exists. Available "camera" names = ('fixed', 'vil_camera', ...)
```
The MuJoCo C API returns -1, but the mujoco_py binding raises -> code like
`cid = model.camera_name2id("top"); if cid >= 0:` is NOT a safe existence check.
Always test `if "top" in [str(n) for n in model.camera_names]` first (see
`diffusion_policy_3d/dr3d/bench/dynamic_hand.py` / `dynamic.py`). pen (DAPG_pen.xml) has **no** 'top'
camera: its policy camera is the oblique `vil_camera` (see `ENV_POINT_CLOUD_CONFIG['adroit_pen']`).

- Error: python -m train ... Fails with PathNotFoundError Due to Incorrect Current Working Directory.
Check this [issue](https://github.com/YanjieZe/3D-Diffusion-Policy/issues/152) for a fix.
