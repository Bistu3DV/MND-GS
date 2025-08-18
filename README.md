# MND-GS: Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction
Bo Jia, Yanan Guo, Ying Chang, Benkui Zhang, Ying Xie, Kangning Du, Lin Cao

For the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multiview normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss.

## Installation

```shell

git clone https://github.com/Bistu3DV/MND-GS.git
cd MNDGS

conda create -n mndgs python=3.8
conda activate mndgs

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Dataset Preprocess
Please download the preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/), the Mip-NeRF 360 dataset from the [official webiste](https://jonbarron.info/mipnerf360/). You need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36). 

The data folder should like this:
```shell
data
├── dtu_dataset
│   ├── dtu
│   │   ├── scan24
│   │   │   ├── images
│   │   │   ├── mask
│   │   │   ├── sparse
│   │   │   ├── cameras_sphere.npz
│   │   │   └── cameras.npz
│   │   └── ...
│   ├── dtu_eval
│   │   ├── Points
│   │   │   └── stl
│   │   └── ObsMask
└── MipNeRF360
    ├── bicycle
    └── ...
```

## Training and Evaluation
```shell

# DTU dataset
python scripts/run_dtu.py

# Mip360 dataset
python scripts/run_mip360.py
```

#### Some Suggestions:
- Adjust the threshold for selecting the nearest frame in ModelParams based on the dataset;
- -r n: Downsample the images by a factor of n to accelerate the training speed;
- --max_abs_split_points 0: For weakly textured scenes, to prevent overfitting in areas with weak textures, we recommend disabling this splitting strategy by setting it to 0;
- --opacity_cull_threshold 0.05: To reduce the number of Gaussian point clouds in a simple way, you can set this threshold.
```shell
# Training
python train.py -s data_path -m out_path --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

#### Some Suggestions:
- Adjust max_depth and voxel_size based on the dataset;
- --use_depth_filter: Enable depth filtering to remove potentially inaccurate depth points using single-view and multi-view techniques. For scenes with floating points or insufficient viewpoints, it is recommended to turn this on.
```shell
# Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01
```

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Densify is based on [AbsGau](https://ty424.github.io/AbsGS.github.io/) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields?tab=readme-ov-file). The single-view normal loss and multi-view consistency constraint are based on [PGSR](https://github.com/zju3dv/PGSR). DTU dataset preprocess are based on [Neuralangelo scripts](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Evaluation scripts for DTU dataset are based on [DTUeval-python](https://github.com/jzhangbs/DTUeval-python). Thank you to all authors for their valuable work and shared repositories.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.
<!-- 
```bibtex
@misc{jia2025multiviewnormaldistanceguidance,
      title={Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction}, 
      author={Bo Jia and Yanan Guo and Ying Chang and Benkui Zhang and Ying Xie and Kangning Du and Lin Cao},
      year={2025},
      eprint={2508.07701},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.07701}, 
}
``` -->
```bibtex
@article{jia2025multi,
  title={Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction},
  author={Jia, Bo and Guo, Yanan and Chang, Ying and Zhang, Benkui and Xie, Ying and Du, Kangning and Cao, Lin},
  journal={arXiv preprint arXiv:2508.07701},
  year={2025}
}
```