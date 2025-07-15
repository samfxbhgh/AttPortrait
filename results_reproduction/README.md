<h1 align="center">Paper Results Reproduction </h1>

Please run on 10.29.150.201
```shell
cd /data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait
```
,then do all the following steps
## 🔥 Training

### Full 
```shell
bash run_train.sh 'train.py' 'paper' 8 '0,1,2,3,4,5,6,7'
```

### w/o Target Attributes
```shell
bash run_train.sh 'train_wo_target.py' 'paper_wo_target' 8 '0,1,2,3,4,5,6,7'
```

### w/o Disentanglement Branch
```shell
bash run_train.sh 'train_wo_disent.py' 'paper_wo_disent' 8 '0,1,2,3,4,5,6,7'
```

### w/o Denoising Branch
```shell
bash run_train.sh 'train_wo_denoise.py' 'paper_wo_denoise' 8 '0,1,2,3,4,5,6,7'
```
### w/o ID Loss
```shell
bash run_train.sh 'train.py' 'paper_wo_ID_loss' 8 '0,1,2,3,4,5,6,7' --coef_id 0
```
### w/o DCA
```shell
bash run_train.sh 'train_wo_DCA.py' 'paper_wo_DCA' 8 '0,1,2,3,4,5,6,7'
```

## :zap: Evaluation

### Full
CelebA-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper' 10 2 "celeba5K" 4 '0,1,2,3' 60 
```

CelebA-test*
```shell
bash test_pipeline_recur.sh 'test.py' 'paper' 2 2 "celeba1K" 4 '0,1,2,3' 60 
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper' 5 2 "syn500" 4 '0,1,2,3' 60 
```

### w/o Target Attributes
CelebA-test*
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_target' 2 2 "celeba1K" 4 '0,1,2,3' 60
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_target' 2 2 "syn500" 4 '0,1,2,3' 60
```
### w/o Disentanglement Branch
CelebA-test*
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_disent' 2 2 "celeba1K" 4 '0,1,2,3' 60
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_disent' 2 2 "syn500" 4 '0,1,2,3' 60
```
### w/o Denoising Branch
CelebA-test*
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_denoise' 2 2 "celeba1K" 4 '0,1,2,3' 60
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_denoise' 2 2 "syn500" 4 '0,1,2,3' 60
```

### w/o ID Loss
CelebA-test*
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_ID_loss' 2 2 "celeba1K" 4 '0,1,2,3' 60
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test.py' 'paper_wo_ID_loss' 2 2 "syn500" 4 '0,1,2,3' 60
```

### w/o DCA
CelebA-test*
```shell
bash test_pipeline_recur.sh 'test_wo_DCA.py' 'paper_wo_DCA' 2 2 "celeba1K" 4 '0,1,2,3' 60
```

Synth-test
```shell
bash test_pipeline_recur.sh 'test_wo_DCA.py' 'paper_wo_DCA' 2 2 "syn500" 4 '0,1,2,3' 60
```

### How to see the FID score
Take the 'paper' folder and the dataset 'celeba-test*' as an example
```text
paper/checkpoint-60000
├── ...
├── celeba1K_FID_random.txt
└── ...
```
The FID score is in the file `celeba1K_FID_random.txt`

## Data Path
### LAION Face
```shell
image dir1 = '/data/hezhenliang/users/zhangyuhao/aligned_5M_size-116_score-0.78_clipiqap-0.385/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/data'
embedding dir1 = '/data/hezhenliang/users/zhangyuhao/aligned_5M_size-116_score-0.78_clipiqap-0.385/analysis/embeddings'
image jsonl1 = '/data/hezhenliang/users/zhangyuhao/data/not_sp.jsonl'

#Restoration
image dir2 = '/data/hezhenliang/users/zhangyuhao/results_v14_572'
embedding dir2 = '/data/hezhenliang/users/zhangyuhao/embeddings'
image jsonl2 = '/data/hezhenliang/users/zhangyuhao/data/filter_sp.jsonl'
```

### CelebA-test (CelebA5K)
```shell
image dir = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/celeba5K'
embedding dir = '/data/hezhenliang/users/zhangyuhao/data/embeddings'
image jsonl = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/celeba5K.jsonl'
```

### CelebA-test* (CelebA1K)
```shell
image dir = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/celeba1K'
embedding dir = '/data/hezhenliang/users/zhangyuhao/data/embeddings'
image jsonl = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/celeba1K.jsonl'
```

### Synth-test (Synth500)
```shell
image dir = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/arc2face_synth-500/aligned/alignment/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)/data'
embedding dir = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/arc2face_synth-500/aligned/analysis/embeddings'
image jsonl = '/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/datas/arc2face_synth-500/myval.jsonl'
```

Please see [attportrait/data.py](../attportrait/data.py) for more details.

## Weights Path
### Our Model
| Variant                   | Path                                                                 |
|--------------------------|----------------------------------------------------------------------|
| **Full Model**           | `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper/checkpoint-60000/unet` |
| **w/o Target Attributes**| `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper_wo_target/checkpoint-60000/unet` |
| **w/o Disentanglement**  | `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper_wo_disent/checkpoint-60000/unet` |
| **w/o Denoising**        | `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper_wo_denoise/checkpoint-60000/unet` |
| **w/o ID Loss**          | `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper_wo_ID_loss/checkpoint-60000/unet` |
| **w/o DCA**              | `/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/paper_wo_DCA/checkpoint-60000/unet` |

### Stable Diffusion v1.5
`/data/hezhenliang/users/zhangyuhao/stable-diffusion-v1-5`

### Other weights
`/data/hezhenliang/users/zhangyuhao/check_paper/AttPortrait/models`

Please see [README.md](../README.md) for more details.