
## Quick Start
### Environment preparation
```bash
conda create -n SGAR python=3.6 -y
conda activate SGAR
# install pytorch according to your cuda version
# don't change version of torch, or it may occur conflict
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge 

pip install -r requirements.txt 
```

### Dataset Preparation

#### 1. Download the COCO train2014 to SGAR/ln_data/images.
```bash
wget https://pjreddie.com/media/files/train2014.zip
```

#### 2. Download the RefCOCO, RefCOCO+, RefCOCOg to SGAR/ln_data.
```bash
mkdir ln_data && cd ln_data
# The original link bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip is no longer valid, we have uploaded it to Google Drive (https://drive.google.com/file/d/1AnNBSL1gc9uG1zcdPIMg4d9e0y4dDSho/view?usp=sharing)
wget 'https://drive.usercontent.google.com/download?id=1AnNBSL1gc9uG1zcdPIMg4d9e0y4dDSho&export=download&authuser=0&confirm=t&uuid=be656478-9669-4b58-ab23-39f196f88c07&at=AN_67v3n4xwkPBdEQ9pMlwonmhrH%3A1729591897703' -O refcoco_all.zip
unzip refcoco_all.zip
```

#### 3. Run data.sh to generate the annotations.
```bash
mkdir dataset && cd dataset
bash data.sh
```

### Training & Testing
```bash
bash train.sh 0,1
bash test.sh 0
```

## Citation
If you find our work useful in your research, please consider citing:
