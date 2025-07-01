<h1 align="center"> AttPortrait </h1>

## 🛠️ Installation

> The code is tested with python 3.10.13, torch 2.0.1, and cuda 11.7.

Clone this repo:

```shell
git clone https://github.com/VIPL-GENUN/Jodi.git
cd Jodi
```

Create and activate a new conda environment:

```shell
conda env create -f environment.yml 
conda activate AttPortrait 
```

## 🤖️ Download Models

We provide our model on [HuggingFace](https://huggingface.co/VIPL-GENUN/Jodi).
The model will be automatically downloaded when you launch the Gradio demo,
or you can download it manually using the following command:

```shell
huggingface-cli download VIPL-GENUN/Jodi
```

## 🔥 Training

### Step 1: Data Preparation

To help you get started quickly, we provide a small example dataset in [assets/example_data](./assets/example_data) with the same file structure as out training dataset.

```text
assets/examples_data
├── metadata.jsonl
├── image
│   ├── 182660.jpg
│   ├── 182676.jpg
│   ├── 182747.jpg
│   └── 182768.jpg
├── embeddings
│   ├── 182660.jpg.embedding.npy
│   ├── 182676.jpg.embedding.npy
│   ├── 182747.jpg.embedding.npy
│   └── 182768.jpg.embedding.npy
└── ...
```

The code will load the data based on `metadata.jsonl`.
Each line in `metadata.jsonl` is a dictionary containing paths to an image and its annotations (attribute labels).
For example:

```python
{"image_name": "182768.jpg", 
"attributes": [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1]}
```

### Step 2: Download Models

To use AttPortrait, you need to download the model files from [Arc2Face](https://github.com/foivospar/Arc2Face) and place them in the "model" directory. 

You also need two additional files:
1. The **attribute classifier checkpoint**
2. The **face recognition model** from [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

For your convenience, we provide a simplified package containing only the required files, which you can download from [Google Drive](https://drive.google.com/drive/folders/1G_49ZzcZW6ZH8M30kbuTIG0uNqR8Zyco?usp=sharing).

After downloading, please place all required files into the `model` directory.  
Make sure the directory structure is as follows:
```text
model
├── antelopev2
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   ├── arcface.onnx
│   ├── genderage.onnx
│   └── scrfd_10g_bnkps.onnx
├── arc2face
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── encoder
│   ├── config.json
|   └── pytorch_model.bin
├── classifier
│   └── best_model.ckpt
├── arcface_glint360K_r100
│   └── backbone.pth
```

### Step 3: Start Training

Training from example dataset:

```shell
bash run_train.sh 'train.py' 'result' 4 '0,1,2,3'
```

## :zap: Quick Inference

```bash
python inference.py
```


## 🪧 Acknowledgement

This project is built upon [Arc2Face](https://github.com/foivospar/Arc2Face). Thanks for their great work!
