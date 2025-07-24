<h1 align="center">Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model</h1>
<p align="center"><a href="https://www.arxiv.org/abs/2507.11465"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://cg.postech.ac.kr/research/Elevate3D'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<p align="center"><img src="Assets/fig_teaser.png" width="100%"></p>

***Check out our [Project Page](https://cg.postech.ac.kr/research/Elevate3D) for videos!***

<!-- Abstract -->
## 📖 Abstract
High-quality 3D assets are essential for various applications in computer graphics and 3D vision but remain scarce due to significant acquisition costs. To address this shortage, we introduce Elevate3D, a novel framework that transforms readily accessible low-quality 3D assets into higher quality. At the core of Elevate3D is HFS-SDEdit, a specialized texture enhancement method that significantly improves texture quality while preserving the appearance and geometry while fixing its degradations. Furthermore, Elevate3D operates in a view-by-view manner, alternating between texture and geometry refinement. Unlike previous methods that have largely overlooked geometry refinement, our framework leverages geometric cues from images refined with HFS-SDEdit by employing state-of-the-art monocular geometry predictors. This approach ensures detailed and accurate geometry that aligns seamlessly with the enhanced texture. Elevate3D outperforms recent competitors by achieving state-of-the-art quality in 3D model refinement, effectively addressing the scarcity of high-quality open-source 3D assets.

<!-- Updates -->
## ⏩ Updates & To Do

**07/23/2025**
- Initial code release.
- [ ] Upload data pre-processing code.


## 🚀 Getting Started

### Prerequisites
- **OS**: Tested only on **Linux**.
- **Hardware**: We recommend using an NVIDIA GPU with at least 48GB of memory due to the requirement of FLUX. The code has been verified on NVIDIA A6000 GPUs.  
- **Software**:   
  - NVIDIA Driver & CUDA Toolkit 12.0 or later.
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) for environment management.
  - Python version 3.10 or higher

### Installation Steps
1. Clone the repo:
    ```sh
    git clone --recurse-submodules https://github.com/ryunuri/Elevate3D.git
    cd Elevate3D
    ```

2. Set Up the Environment:
    Create and activate the `elevate3d` conda environment using the provided file.
    ```sh
    conda env create -f environment.yml --name elevate3d
    conda activate elevate3d
    ```
3. Download Models & Dependencies:
    You need to download pre-trained models and build one external dependency.

    __A. Download Checkpoints__:
    Our framework relies on several off-the-shelf models. Some will be downloaded automatically from Hugging Face, but others need to be placed manually.

    ```sh
    # Create directories for checkpoints
    mkdir -p Checkpoints/sam
    
    # Download the Segment Anything Model (SAM) checkpoint
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P Checkpoints/sam/
    ```

    __B. Build PoissonRecon__:
    The geometry refinement step uses Poisson Surface Reconstruction. You need to build the executable from the source.

    ```sh
    # Clone the PoissonRecon repository
    git clone https://github.com/mkazhdan/PoissonRecon.git
    
    # Navigate and build the executable
    cd PoissonRecon/
    make
    cd ..
    ```

<!-- Usage -->
## 🎮 Usage
Before running, make sure you have downloaded the necessary example data and configured your .yaml file with the correct paths to the checkpoints and PoissonRecon executable.

### 📥 Download Example Data

Before running the examples, you need to download the sample 2D images and low-quality 3D models.

1.  **Download the file:** Click the link below to download `Inputs.zip` from Google Drive.
    *   [**Download Example Data (Inputs.zip)**](https://drive.google.com/file/d/1VJmnT2UQKsYuZijGeAuJ0fOeOqkP8nya/view?usp=sharing)

2.  **Unzip the file:** Move the downloaded `Inputs.zip` to the root of this project directory (e.g., `Elevate3D/`). Then, run the following command in your terminal to create an `./Inputs` folder and extract the files into it:

    ```bash
    # Make sure Inputs.zip is in the current project directory
    unzip Inputs.zip
    ```
Before running the examples the directory structure should be like this:

```
Elevate3D/
├── Checkpoints/                <-- For pre-trained models
│   └── sam/
│       └── sam_vit_h_4b8939.pth
├── Inputs/                     <-- Example data you just downloaded
│   ├── 2D/
│   └── 3D/
├── PoissonRecon/               <-- For geometry processing
│   └── Bin/
│       └── Linux/
│           └── PoissonRecon    <-- The compiled executable
├── ... (other project files)
└── README.md
```

### Example 1: 2D Image Refinement (HFS-SDEdit)
An example of using HFS-SDEdit for 2D image refinement.

This runs our texture enhancement module on a single image.

```python
python -m FLUX.flux_HFS-SDEdit
```

### Example 2: Full 3D Model Refinement
This script runs the complete Elevate3D pipeline on an example model. It will perform iterative texture and geometry refinement.

```sh
bash run_3d_refine_script_example.sh
```

## 🙏 Acknowledgements

This work builds upon the fantastic research and open-source contributions from the community.
We extend our sincere thanks to the authors of the following projects:

- [FLUX](https://github.com/black-forest-labs/flux)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Marigold](https://github.com/prs-eth/Marigold)
- [Marigold E2E](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)
- [PoissonRecon](https://github.com/mkazhdan/PoissonRecon)
- [continuous-remeshing](https://github.com/Profactor/continuous-remeshing)
- [InTeX](https://github.com/ashawkey/InTeX)

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3721238.3730701,
author = {Ryu, Nuri and Won, Jiyun and Son, Jooeun and Gong, Minsu and Lee, Joo-Haeng and Cho, Sunghyun},
title = {Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model},
year = {2025},
isbn = {9798400715402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3721238.3730701},
doi = {10.1145/3721238.3730701},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {165},
numpages = {12},
keywords = {3D Asset Refinement, Diffusion models},
location = {},
series = {SIGGRAPH Conference Papers '25}
}
```
