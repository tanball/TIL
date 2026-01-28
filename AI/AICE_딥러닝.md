# AI

> 출처 : AICE associate 강의를 보고 작성하였습니다.

## 딥러닝

### 딥러닝 개념, 기술원리 및 주요 알고리즘


**딥러닝 기본 개념**

데이터의 특징을 파악하고 패턴을 인식하여 학습함.

목표 : 모델에 입력값을 넣었을 때의 출력값이 최대한 정답과 일치하게 하는 것.

딥러닝 학습 방법 : 

딥러닝 모델의 매개변수(weight, bias)를 무작위로 부여한 후 반복학습을 통해 모델의 출력값을 정답과 일치하도록 매개변수를 조금씩 조정함

-> Gradient Descent(경사 하강) 최적화 알고리즘 이용


**딥러닝 기술 원리**

Perceptron : 사람 뇌에 있는 뉴런을 모델링한 것. 간단한 함수를 학습할 수 있음.

입력 \( x_n \) → 가중치 \( w_n \) → \( \sum \) → 활성함수 \( y = f(w^T x) \)


Inputs   ->  Weight.  ->     -> Activation Function

Logistic Regression Model과 유사한 면도 있다


DNN(Deep Neural Network) : 심층신경망. 입력층과 출력층 사이에 여러개의 은닉층(hidden layer)으로 이루어진 인공신경망. 신경망 출력에 비선형 활성화 함수를 추가해 복잡한 비선형 관계를 모델링 할 수 있음.

입력층 -> 은닉층1 -> … -> 출력층
 
은닉층의 개수는 정해지지 않았다.

각 원 = 노드 = 퍼셉트론(perceptron) = 유닛

> 참고로 input layer는 그냥 입력.

> 즉, 입력층이 x값들을 입력. 은닉층의 각 노드는 x를 받아 w로 바꾸고 sum하여 활성함수를 돌리고 결과를 다음 은닉층 노드의 x로 보낸다.


Activation function : 활성화 함수. 입력 신호의 총합을 출력 신호로 변환하는 함수. 입력 신호의 총합이 활성화를 일으키는가? Yes->1, No->0. 선형 함수가 아니라 비선형 함수가 필요.

* Binary Step

$$
f(x) =
\begin{cases}
0, & x < 0 \\
1, & x \ge 0
\end{cases}
$$

* Logistic, sigmoid, or soft step

$$
f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

* Hyperbolic tangent (tanh)

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

* Rectified linear unit (ReLU)

$$
f(x) =
\begin{cases}
0, & x < 0 \\
x, & x \ge 0
\end{cases}
= \max\{0, x\}
$$


**활성화 함수(Activation functioin) 설명**

- Binary Step : 0과 1로 구분되는 계단함수.
- Logistic(혹은 Sigmoid. 같은 뜻) : 혹은 soft step. 0초과 1미만 값을 갖는 연속형 함수.
- Hyperbolic tangent : 가로로 누운 탄젠트 모양? -1초과 1미만 값을 갖는 연속형 함수.
- ReLU : 0 이하는 0, 그 이상은 그 값 그대로 출력.
- ReLU 강화 버전인 SELU, LReLU, ELU 등(-값을 조금 보전하는 형태)
- Softmax : 많이 쓰이는 activation function중 하나. 특히 분류에서 많이 사용.
    분류의 결과가 1, 2, 3 중 하나가 될 때, 1일 확률, 2일 확률, 3일 확률 각각을 결과로 준다. 그리하여 모든 가짓수의 값을 합치면 1이 된다.
    이전의 linear함수로 3개의 값을 얻었다면, 이 값이 소프트맥스 함수를 거쳐 각각 3개의 확률값(다 합하면 1)으로 바뀜. 그리고 이 중 가장 큰 확률에 해당한다고 예츠.
    수식:
	$$
    \sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}, \quad \text{for } j = 1, \dots, K.
    $$



**Loss Function**

손실함수. 정답과 예측값 사이의 차이를 계산한다. 이 값이 최소가 되도록 하는 것이 학습의 궁극적인 목적. 신경망 학습의 목적함수.

종류:

회귀(Regression)

- MSE

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- MAE

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

분류(Classification)

- 이진분류 (Binary cross-entropy)

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[ t_i \log(y_i) + (1 - t_i)\log(1 - y_i) \right]
$$

- 다중분류 (Categorical cross-entropy)

$$
L = -\frac{1}{N} \sum_{j=1}^{N} \sum_{i=1}^{C} \left[ t_{ij} \log(y_{ij}) \right]
$$

분류의 손실함수는 cross-entropy이다.  
이 때, 2가지로 나뉘는가, 3가지 이상으로 나뉘는가에 따라 명칭이 달라진다.
**손실함수 용어 기억하기**


**Optimization(최적화)**

딥러닝 모델의 매개변수를 조절해 손실함수 값을 최저로 만드는 과정.

Gradient Descent(경사하강법)이 대표적

DNN모델의 가중치 파라미터들을 최적화하는 방법

손실함수의 기울기(Gradient)를 이용 -> Loss 최저점 = 기울기 0


**역전파(Back Propagation)**

실제값과 모델 결과값에서 오차를 구해서 오차를 output에서 input방향으로 보냄.->가중치를 재업데이트하며 학습
 

*순전파(Forward Propagation)* : 딥러닝 모델에 값을 입력하여 출력을 얻는 과정

    입력이 들어오고, 모델을 통과해 결과를 내고, 점수를 확인

*역전파(Error Back Propagation)* : 정답과 모델 결과값에서 오차를 구해 오차를 output 레이어에서 input 레이어 방향으로 보내며 가중치를 재업데이트하는 과정.

    순전파로 얻은 예측값과 정답을 비교해 오차(Error)를 얻고, output layer에서 input layer방향으로 gradient descent 계산하며 가중치 업데이트를 단계적으로 진행.


**최적화 알고리즘들(Optimiztion Algorithm)**

GD : Gradient Descent Algorithm. 학습 너무 오래걸림

> 데이터가 100만개라면, 순전파(100만)-역전파(1) 이 1회 학습.

SGD : Stochastic GD. 전체 데이터를 배치사이즈로 나누어 배치사이즈마다 가중치 업데이트

> 100만개 데이터에 10만개 배치사이즈라면 순전파(10만)-역전파(1) *10 이 1회 학습

이 외에도 여러가지 있으며, 우리는 Adam 사용하면 됨.


**추가 용어**

Data Set 관련.

batch size : 데이터셋을 여러개로 쪼갠 하나. 보통 1/10

1 Epoch : 모든 데이터셋을 한 번 학습. 10*batch_size

*SGD가 batch size마다 순전파/역전파 학습->가중치 업데이트


**Dropout**

과적합을 막기 위해, Train 학습 시에만 히든 레이어의 일부 노드를 빼고(Drop) 동작하지 않게 한 채 학습시키는 방법.


**코딩 템플릿**
```
import tensorflow as tf
from tensorflow.keras.models import Sequential 
#히든 레이어를 담을 Sequential함수를 임포트
from tensorflow.keras.layers import Dense, Dropout 
#히든 레이어인 Dense와 오버피팅 방지용 Dropout 임포트

#DNN 예시 이미지와 노드 배치가 동일하게 모델 생성
model=Sequential() #모델 정의
model.add(Dense(4, imput_shape=(3,),activation=’relu’)) #정의한 모델에 내용 추가.
#첫 번째 히든 레이어. 4개 유닛을 갖는다. 첫 번째 히든 레이어니 인풋을 받는다. 3개 인풋. relu함수.
model.add(Dropout(0.2)) #필요에 따라 이렇게 Dropout을 추가. 0.2(20%)확률로 dropout
model.add(Dense(4, activation=’relu’)) #두 번째 히든 레이어. 4개 유닛. relu 함수.
model.add(Dense(1, activation=’sigmoid’)) 
#세 번째 히든레이어. 마지막이기에 output layer가 된다. 출력이니 노드 하나. sigmoid 함수로 0과 1 사이로 변환.

model.compile(loss=’binary_crossentropy’, opitmaizer=’adam’, metrics=[‘accuracy’])
#모델 컴파일. loss function은 분류 모델이니 crossentropy(2개로 분류). adam 쓰고, 한 번 학습시마다 정확도(accuracy)측정.
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=10)
#지도 학습이니 x y train 데이터를 줘서 fit. x y test데이터를 검증 데이터로 이용. 전체 데이터를 40번 학습하고(epochs), 전체 데이터를 10으로 나눈 크기를 하나의 batch 사이즈로 한다. 이를 가지고 모델을 학습시켜 결과를 history에 저장한다.
```
sequential>dense>compile>fit 순으로 진행.
