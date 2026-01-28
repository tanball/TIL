# AI

> 출처 : AICE associate 강의를 보고 작성하였습니다.

## 머신러닝

### 머신러닝 개념과 기술원리


**머신러닝 기본 개념**

과거 : 사람이 데이터 패턴 찾아 알고리즘 코딩해 결과 얻음

AI/머신러닝 : 데이터와 결과 기반으로 스스로 패턴 학습해 이를 이용해 예측


**Linear Regression(선형 회귀)**

직선을 그어서 예측하는 모델.

> 가설 : 공부를 많이 하면 공부를 잘할 것이다.
>
> 토익점수와 학습시간의 관계 산점도
>
> 해당 데이터에 가장 잘 맞는 직선 구하기

직선의 방정식 : *ŷ = wx + b*

‘가장 잘 맞는 직선’ : 산점도의 점들을 가장 잘 표현하는 최적의 직선(회귀선)

최적 판단 기준 : Cost Function

**Cost Function** *= (∑(실제값-예측값)^2 )⁄N -> MSE(Mean Squred Error)*

> 실제 점(20, 600)과 예측값(20,400)의 차이(200)의 제곱을 데이터 샘플 개수로 나누는 것을 cost로 하는 방식이 MSE(평균 제곱 오차)


**Cost Function의 최적화**

Gradient Descent Algorithm(경사 하강법) 이용

회귀선은 직선 y=wx+b

MSE는 2차함수의 포물선 형태(x=w, y=cost)가 되며, 

cost가 최소가 되는 w의 값을 구하여 회귀선을 구한다.

*W≔W - {α * ∂/∂W cost(W)}*

α  : 학습율,  ∂/∂W : 미분

(새로운 W) = (이전에 구한 W) – 학습율 * 미분 기울기

**경사 하강법** : cost function 함수(x대신 W, y대신 cost)의 기울기를 구하여(미분) 그 기울기가 최소가 되는 w를 구한다.


**모델 학습 개념 정리**

목표 : 최적 직선 구하기

직선별 손실함수 구하기

손실함수 최소값 구하기

Gradient Descent Algorithm 이용

*물론 이 단계는 모델 내에서 자동 진행


**머신러닝 기술 원리**


**데이터 확보**

잘 정리된 데이터 확보가 중요

실제로는 무의미한 데이터, 편향된 데이터 등 다양한 문제가 발생.	


**학습방법 종류**
지도학습(supervised learning)

> 정답을 알려주며 진행하는 학습
>
> 데이터와 레이블(정답) 함께 제공
>
> *레이블(Label) = 정답, 실제값, 타깃, 클래스, y
>
> *예측된 값 = 예측값, ŷ(y hat)

비지도학습(un-supervised learning)

> 레이블(정답)없이 진행되는 학습
>
> 데이터 자체에서 패턴을 찾아야 할 때 사용


**지도학습 모델 종류**
분류모델(classification)

>레이블의 값들이 이산적으로 나누어 질 수 있는 문제에 사용
>
>> ex)남/여 분류, 학점 A/B/C/D/E 분류 등
예측모델(regression, =회귀모델)
>
> 레이블 값들이 연속적인 문제에 사용
>
>> ex)팁 값 예측, 삼전 주가 예측 등


**데이터셋 분리**

전체 데이터셋 -> Train_set / Test_set 혹은 Train_set / Validation_set / Test_set 으로 나눈다

Train 데이터셋으로 모델을 학습시킨다(참고서)

Validation 데이터셋(검증 데이터셋)으로 학습시마다 모델의 성능을 확인하고(모의고사)

Test 데이터셋으로 최종적인 모델 성능 평가를 수행.(수능시험)


**과적합(OverFitting) 문제**

- 예측값과 정답의 오차가 거의 없다.

- Train 데이터에만 지나치게 맞추어져서 Test 데이터로 확인하면 점수가 나쁨

**과소적합(Underfitting)**

- 보통 제대로 학습되지 않아 발생. 더 학습시키면 된다.

   


**모델 성능 평가**

학습 횟수에 따른 오차(성능)그래프 : 

x축 Epochs(반복횟수), y축 Error(오차)인 그래프에서 트레이닝 셋과 테스트 셋 모두 Rational 그래프(1/x)에 가까운 형상으로 그려짐.

즉, 처음에는 반복횟수 증가 시 빠르게 오차가 감소하며, 이후에는 매우 미미하지만 꾸준히 오차가 감소.

*이 때, 반복횟수가 증가할 때, 트레이닝셋과 달리 테스트셋의 에러가 오히려 증가한다면 -> 과적합


**성능지표**

학습이 끝나고 모델을 성능평가하는 용도로 사용된다.

- 회귀모델 성능지표 : MSE, MAE, R2

- 분류모델 성능지표 : 정확도, 정밀도, 재현율, F1-점수

**회귀 성능지표**

MSE

> Mean Squared Error : 평균 제곱 오차
>
> 예측값에 대한 실제값의 오차를 구하고 그 제곱값의 평균을 구하는 방식
>
> *MSE = ∑{(y-ŷ)^2} / n*
>
> *이상치 데이터가 존재하면, 제곱을 하기에 MAE보다 더 크게 영향을 받음

MAE

> Mean Absolute Error : 평균 절댓값 오차
>
> 예측값에 대한 실제값의 오차를 구하고 그 절댓값 평균을 구하는 방식
>
> *MAE = ∑|y-ŷ| / n*

R2

> 결정계수
>
> 회귀모델에서 독립변수가 종속변수를 얼마나 잘 설명해주는지 나타내는 지표
>
> R2스코어가 1에 가까울수록 좋은 모델(MSE, MAE는 0에 가까워야 좋다)
>
> *R^2 = 1 - ∑{(t-y)^2} / ∑{(t-t_bar)^2} *
>
> t = 실제값, y = 예측값,  t bar = 평균값
>
> 1-(실제값과 예측값의 오차 제곱의 총합)/(실제값과 그 평균의 차 제곱의 총합)


**분류모델 성능지표**

Confusion Matrix

오차 행렬

|  | 예측(0) | 예측(1)
| :--- | :---: | :---: |
| **실제(0)** | TN | FP |
| **실제(1)** | FN | TP | 
 
TN : 예측이 맞았으며(True), 내 예측값은 False(Negative)다

FP : 예측은 틀렸고(False), 내 예측값은 True(Positive)다

FN : 예측은 틀렸고, 내 예측값은 False다

TP : 예측이 맞았고, 내 예측값은 True다.

*답을 예측할 때, True(0)/False(1)값으로 예측. 

특정 지표는 아니며, 지표를 검증하는데 쓰는 행렬.


- 정확도(Accuracy)

> 가장 직관적으로 모델의 성능을 나타낼 수 있는 평가지표
>
> (Accuracy)=  (TP+TN)/(TP+FN+FP+TN) 
>
> 모든 오차행렬의 값을 더하고, 그 중 True인 것만을 더한다.
>
>> 정확도 함정 : 극단적으로 암환자 1명에 건강인 99명을 대상으로 테스트했을 때, 무조건 ‘암 환자가 
아니다’라고 진단하면 99%의 정확도가 나옴. -> 이럴 때는 다른 성능지표도 함께 봐야 함.
>
>> 100개 데이터로 테스트하여, TP=30, FN=20, FP=10, TN=40 이면->70/100=70%



- 정밀도(Precision)

> 모델이 True라고 분류한 것 중, 실제 True인(정답인) 비율.
>
> *(Precision)=  TP/(TP+FP)*
>
>> 암환자 분류 사례 같은 경우에 사용

- 재현율(Recall)

> 실제 True인 것 중에 모델이 True라고 예측한 것의 비율
>
> *(Recall)=  TP/(TP+FN)*
>
>> 스팸 분류 같은 경우에 사용

- F1점수

> 정밀도와 재현율을 섞은 것(조화평균)
>
> *(F1-score)=2×1/(1/Precision+1/Recall)*




### 머신러닝 주요 알고리즘


**Scikit-learn**

가장 인기 있는 머신러닝 패키지, 많은 머신러닝 알고리즘 내장됨.

원하는 모델을 import하고, 사용한다.

> from sklearn.family import Model

> from sklearn.linear_model import LinearRegression	#LinearRegression모델을 import
>
> model=Linear Regression()		#임포트한 모델 사용.


**머신러닝 주요 알고리즘 분류**

회귀

> Linear Regression

분류

> Logistic Regression

회귀분류

> Decision Tree
>
> Random Forest
>
> K-Nearest Neighbor


**Linear Regression(선형 회귀)**

실행 방법

> from sklearn.linear_model import LinearRegression #원하는 모델 임포트
>
> model = LinearRegression() #모델 정의
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #예측이나 성능을 확인(pred는 정답)


**Logistic Regression(논리 회귀)**

이진 분류 규칙은 0, 1의 두 클래스를 갖는 것이다. 일반 선형 회귀 모델 사용은 어려움.

-> Logistic 함수로 결과값을 0~1사이의 값으로 변환하여 이진 분류

로지스틱 함수
$$
\hat{p} = h_\theta(x) = \sigma(X^T\theta)
\sigma(t) = \frac{1}{1 + e^{-t}}
\hat{y} = \begin{cases} 
0, & \hat{p} < 0.5 \\
1, & \hat{p} \ge 0.5 
\end{cases}
$$

각 항의 의미
- 로지스틱 함수
*   `$\hat{p}$`: 예측 확률
*   `$h_\theta(x)$`: 가설 함수
*   `$\sigma$`: 시그모이드 함수
*   `$X^T\theta$`: 선형 결합
- 시그모이드 함수
*   `$\sigma(t)$`: 입력 $t$에 대한 함수의 출력값 (0에서 1 사이의 확률값)
*   `$t$`: 함수에 입력되는 변수 또는 값
*   `$e$`: 자연 상수 (약 2.71828)
- 결정 규칙
*   `$\hat{y}$`: 최종 예측 클래스 레이블 (0 또는 1)
*   `$\hat{p}$`: 모델 예측 확률값 (0.5가 기본 임계값)

 
p의 값은 그래프의 y와 같다. p값이 0.5보다 크면 1로, 작으면 0으로 본다.

> ```
> from sklearn.linear_model import LogisticrRegression #원하는 모델 임포트
>
> model = LogisticRegression() #모델 정의
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #정답과 맞춰보며 모델의 학습능력을 확인한다.
> ```


**K-Nearest Neighbor**

새로운 데이터가 주어졌을 때, 기존 데이터 가운데 가장 가까운 k개 이웃의 정보로 새로운 데이터를 예측하는 방법론

> 데이터의 산점도를 그리고, 새 점(데이터)를 그려넣는다. 그 점에서 가장 가까운 k개 점 중 A인 것이 2개, B인 것이 1개면 새 점은 A일 가능성이 높다.

알고리즘이 간단하며, 큰 데이터셋이나 고차원 데이터셋에는 부적합.

> ```
> from sklearn.neighbors import KNeighborsClassifier #원하는 모델 임포트
>
> model = KNeighborsClassifier(n_neighbors=3) #모델 정의. k를 몇으로 할지 지정
>
> # 하이퍼파라미터 : 모델에게 주어지는 정보, 설정값. 모델이 학습하는 파라미터(w,b)와 달리 모델에게 우리가 주고 시작.
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #예측이나 성능을 확인(pred는 정답)
> ```


**Decision Tree**

분류와 회귀 작업이 모두 가능한 다재다능한 머신러닝 알고리즘.

복잡한 데이터셋도 학습 가능

강력한 머신러닝 알고리즘인 랜덤 포레스트의 기본 구성요소

> 붓꽃 예시 : 꽃잎 길이가 2.45보다 작다면 ‘세토사’종, 아니라면 꽃잎 길이가 1.75 이하면 종 B, 아니면 C…

과거에 많이 사용. -> 모델의 예측의 이유를 설명할 수 있기 때문에(다른 모델은 대개 블랙박스)

> ```
> from sklearn.tree import DecisionTreeClassifier #원하는 모델 임포트
>
> model = DecisionTreeClassifier (max_depth=2) #모델 정의. 트리 뎁스를 하이퍼파라미터로.
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #예측이나 성능을 확인(pred는 정답)
> ```


**Random Forest**

일련의 예측기(분류/회귀모델)로부터 예측을 수집하면 가장 좋은 모델 하나보다 더 좋은예측을 얻을 수 있음

‘일련의 예측기’이용 -> 앙상블

‘결정 트리’를 일련의 예측기로 한 앙상블 -> 랜덤 포레스트

훈련 세트로부터 무작위로 각기 다른 서브셋(여러개의 디시전 트리)을 만들어 일련의 결정 트리 분류기를 훈련시킬 수 있음.

> ```
> from sklearn.ensembleimport RandomForestClassifier #원하는 모델 임포트
>
> model = RandomForestClassifier(n_estimators=50) #모델 정의. 생성 트리 수 하이퍼파라미터.
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #예측이나 성능을 확인(pred는 정답)
> ```

랜덤포레스트 하이퍼파라미터 종류

* n_estimators : 생성할 트리의 개수(int, default=100)
* max_depth : 트리의 최대 깊이(int, default=None)
* min_samples_split : 노드 분할에 필요한 최소 샘플 개수(int or float, default=2)

랜덤포레스트 변수중요도

> feature_importance_ : 변수에 대한 중요도 값을 제공


**GridSearchCV**

모델이 아니라, 성능 최적화를 위한 함수다.

하이퍼파라미터를 순차적으로 입력해 학습하고 측정하며 가장 좋은 최적의 파라미터를 알려준다.

주어진 하이퍼파라미터 조합으로 여러 모델을 만들고, 그 중 어느 값들로 한 경우가 가장 좋은지 확인해줌.

많은 경우의 수로 만들어보면 시간 오래 걸릴 수도.

> ```
> from sklearn.model_selection import GridSearchCV #원하는 모델 임포트
>
> rfc= RandomForestClassifier () #모델 정의. 아직 하이퍼파라미터 없음
>
> params = {‘n_estimators’:[100,150], ‘max_depth’:[2,5]} #어떤 파라미터에, 어느 경우의 수를 가지고 할 것인가. 트리 개수는 100개거나, 150개. 트리 깊이는 2거나 5.
>
> grid_rfc = GridSearchCV(rfc, param_gric=params) #지정된 내용으로 rfc모델들 돌려보고 가장 좋았던 파라미터 결과를 알려줄 것.
>
> model.fit(X_train, y_train) #정의된 모델을 학습(fit)시킨다
>
> pred = model.predict(X_test) #예측이나 성능을 확인(pred는 정답)
> ```

그리드서치CV 중요속성

* .best_params_ : 최적의 파라미터 리스트
* .best_score_ : 최적 파라미터일 때의 점수값

### 머신러닝 모델링 실습
앙상블 기법
**앙상블 기법의 종류**
- 배깅 (Bagging): 여러개의 DecisionTree 활용하고 샘플 중복 생성을 통해 결과 도출. RandomForest
- 부스팅 (Boosting): 약한 학습기를 순차적으로 학습을 하되, 이전 학습에 대하여 잘못 예측된 데이터에 가중치를 부여해 오차를 보완해 나가는 방식. XGBoost, LGBM
- 스태킹 (Stacking): 여러 모델을 기반으로 예측된 결과를 통해 Final 학습기(meta 모델)이 다시 한번 예측
