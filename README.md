## Environment
Ubuntu 22.04 

GPU>=RTX3060

Python 3.10.12

## Important
To run `vo_loftr.py`, you need to patch two files in your `kornia` package.
Replace `site-packages/kornia/feature/loftr/loftr.py` with our `loftr/loftr.py`.
Replace `site-packages/kornia/feature/loftr/utils/coarse_matching.py` with our `loftr/coarse_matching.py`.
The patching will change the api of `kornia` package to obtain the feature descriptor of LoFTR. Make sure you modify the module files in a disposable isolated python environment.


## Run
### Prepare
```
virtualenv env
```


```
source ./env/bin/activate
```

```
pip install -r requirements.txt
cd LightGlue
python3 -m pip install -e .
cd ..
```

### LightGlue

#### KITTI
```
python3 vo_lightglue.py dataset2 --camera_parameters camera_params.npy
```

#### NTU
```
python3 vo_lightglue.py frames --camera_parameters camera_parameters.npy
```

### SuperGlue

#### KITTI
```
python3 vo_superglue.py dataset2 --camera_parameters camera_params.npy
```

#### NTU
```
python3 vo_superglue.py frames --camera_parameters camera_parameters.npy
```

### LOFTR

#### KITTI
```
python3 vo_loftr.py dataset2 --camera_parameters camera_params.npy
```

#### NTU
```
python3 vo_loftr.py frames --camera_parameters camera_parameters.npy
```

## View result

```
python3 show_camera.py
```


The name of the plt file needs to be changed to that of the file you want to see.
