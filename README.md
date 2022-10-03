# Face Detection

## Nvidia

## Cuda
To check cuda version need to add the following command
```
nvidia-smi
```

Output should looks like the following snippet
```
Tue Sep 20 21:31:43 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 46%   34C    P0    54W / 250W |    301MiB /  4096MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2923      G   /usr/lib/xorg/Xorg                139MiB |
|    0   N/A  N/A      3176      G   /usr/bin/gnome-shell               31MiB |
|    0   N/A  N/A      5697      G   ...298359942186510770,131072      120MiB |
|    0   N/A  N/A     12683      G   ...RendererForSitePerProcess        2MiB |
+-----------------------------------------------------------------------------+
```

Also need to check the following command
```
nvcc --version
```

Output should looks like the following snippet
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

## Python Packages
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```