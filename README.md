# MCC

## Contents
* [Preparation](#preparation)
* [Run the Code](#run-the-code)
* [Visualization](#visualization)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Preparation
### 1. Download datasets
In our project, the following datasets are used.
Please visit the following links to download datasets:

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

* [CARPK](https://lafi.github.io/LPN/)

* [PUCPR+](https://lafi.github.io/LPN/)

* [IOCfish5k](https://github.com/GuoleiSun/Indiscernible-Object-Counting)
  
We use CARPK and PUCPR+ by importing the hub package. Please click [here](https://datasets.activeloop.ai/docs/ml/datasets/carpk-dataset/) for more information.
```
/
├─MCC/
│
├─FSC147/    
│  ├─gt/
│  ├─image/
│  ├─ImageClasses_FSC147.txt
│  ├─Train_Test_Val_FSC_147.json
│  ├─annotation_FSC147_384.json
│  
├─IOCfish5k/
│  ├─annotations/
│  ├─images/
│  ├─test_id.txt/
│  ├─train_id.txt/
│  ├─val_id.txt/
```


### 2. Download required Python packages:

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install hub
```

If you want to use the docker environment, please download the docker image through the command below
```
docker pull sgkang0305/vlcounter
```

### 3. Download CLIP weight and Byte pair encoding (BPE) file

Please download the [CLIP pretrained weight](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) (or Google Drive link [here](https://drive.google.com/drive/folders/1EwJYQXpC5tZ4D3dXoXCrkBVEPEyWlawl?usp=sharing))and locate the file under the "pretrain" folder.


## Run the Code

### Train
You can train the model using the following command. Make sure to check the options on the train.sh file.
```
python -m tools.train --config config_files/FSC.yaml --gpus 0 --exp 1 --enc spt --num_tokens 10 --patch_size 16 --prompt plural --con rank
```     


### Evaluation
You can test the performance of trained ckpt with the following command. Make sure to check the options in the test.sh file. Especially '--ckpt_used' to specify the specific weight file.
```
python -m tools.test --config config_files/FSC.yaml --gpus 0 --exp 0 --enc spt --num_tokens 10 --patch_size 16 --prompt plural --ckpt_used 'D:/CSAM/VLCounter/pretrain/182_best'
```




## Acknowledgements

This project is based on implementation from [CounTR](https://github.com/Verg-Avesta/CounTR).
