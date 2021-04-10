# pstage_02_image_classification

## Getting Started    
python3 train.py --EPOCHS 10 --BATCH_SIZE 64 --LEARNING_RATE 0.001 --MODEL 'resnet50' --TRAIN_KEY 'classifier' --LOSS 'cross_entropy'

### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### 
config.py
---
args를 불러올 수 있는 파일

dataset.py
---
Train과 eval 데이터 셋 클래스를 구현하였고 각각의 데이터 로더를 불러 올 수 있다

ensemble.py
---
/checkpoint/vote_list에 존재하는 모델들의 soft voting한 결과를 csv로 만들어주는 파일

gpu.py
---
gpu 관련한 자원 확인 및 실험파일

inference.py
---
단일 모델의 결과를 csv로 만드는 파일

loss.py
---
cross entropy, focal, f1, label smoothing의 함수가 있는 파일

metrics.py
---
val 데이터를 기준으로 f1, precision, recall, accuracy 점수를 얻을 수 있는 파일

model.py
---
pretrained 모델을 가져올 수 있는 파일

train.py
---
train 데이터를 통한 모델 학습과 val 데이터를 통한 검증이 이루어지고 검증을 통한 f1 점수를 기준으로 model의 저장이 되는 파일 (+ 추가적으로 wandb에 정보를 보내 시각화 해준다)
