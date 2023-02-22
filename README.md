# 논문 제목 : 인간 운동 제어의 과제 중심 심층 강화 학습 모델을 이용한 fMRI 연구

## 연구 배경
<br>

1. 인공지능의 유래는 뇌 과학이며, 핵심 알고리즘은(i.e, Convolutional Neural Network, Long Short Term Memory, etc..) 인간의 뇌로부터 영감을 받아 고안되었다.
<br>

2. 기존의 뇌과학 연구는 해석의 용이성을 위해 단순한 연구를 진행하였으나, 이는 실제 환경과 차이가 존재했다. 따라서, 보다 실제환경에 가까운 복잡한 연구를 진행하기 시작했고, 이 연구를 위해 **딥러닝 모델**을 적용하기 시작했다.
<br>

## 연구 개요 및 목적
<br>

1. 최근 논문에(Cross et al.,2021) 따르면, end-to-end 방식으로 학습된 Deep Q Network 모델의 파라미터가 인간의 뇌 활동을 일부 설명함을 보였다.
<br>

2. 하지만, 위 논문은 의사결정 분야에 관한 논문으로 연속적인 행동이 아닌 버튼을 클릭하는 **이산적인 행동방식**으로 실험을 진행하였다.
<br>

3. 따라서, 본 프로젝트는 연속적인 운동학습이 심층 강화학습 (Deep Reinforcement Learning; DRL)에 의해 설명될 수 있는지 알아보고자 연구를 진행하였다.
<br>

> "Cross et al.,2021" 과의 차이점<br>
> - 의사결정이 아닌 연속적인 운동학습을 연구하였다.
> - "Cross et al.,2021"과는 달리 사람의 Action Policy와 DRL 모델의 Action Policy간 상관관계를 이용하여 그 근거를 찾고자 하였다.
<br>

## 실험 설계 및 방법
<br>

### 1. fMRI 실험
<br>

- 사람의 뇌 활동을 수집하기 위해, fMRI 장비를 이용하였다.
- 실험은 조이스틱을 이용하여 타겟을 추적하는 간단한 실험으로 진행하였다.
<br><br>

<img src="./img/experiment.jpg" width="350" height="200"/>
