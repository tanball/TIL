# AI

> 출처 : AICE associate 강의를 보고 작성하였습니다.

## Pandas 이해 및 활용


### Pandas 이해 및 활용 – DataFrame 살펴보기


**Pandas**

데이터 분석과 처리에 많이 쓰는 라이브러리.

**DataFrame**

행과 열로 구성된 데이터 구조
[img 첨부]
> 인덱스 : 첫 번째 열. 각 행을 식별하기 위한 레이블.
>
> 컬럼 : 각 열의 이름. 데이터의 변수. 각 컬럼이 하나의 시리즈 객체.
>
> 레코드 : 하나의 행. 동일 인덱스를 갖는 컬럼 값들의 묶음.
>
> 시리즈 : 값+인덱스 구조. 한 컬럼을 단독으로 표현한 형태.
>
> 데이터프레임 : 여러 시리즈가 모인 2차원 구조


**데이터프레임 생성**

1.	딕셔너리 이용

2.	리스트 이용

3.	파일 읽기(pandas 함수 이용)

단계

Import pandas
```python
A1 = pd.dataFrame({“a”:[1,2,3],”b”:[4,5,6],”c”:[7,8,9]})#딕셔너리로 데이터프레임 생성
A2 = pd.dataFrame([[1,2,3],[4,5,6],[7,8,9],columns=[‘a’,’b’,’c’]])#리스트로 
```

데이터프레임 생성

A3 = pd.read_csv(‘file1.csv’) #csv 파일 읽어 데이터프레임 생성

데이터프레임의 함수
> df.head() : 앞5개 라인 출력
>
> df.tail() : 뒤 5개 라인 출력
>
> df.shape : 행, 열 개수 튜플로 반환
>
> columns : 컬럼명 확인 가능
>
> info : 데이터 타입, 각 아이템 개수 출력
>
> describe : 데이터 요약 통계량을 나타냄

**데이터 조회**

1.	조회할 컬럼 선택하기

>	[]을 이용해 column 추출
>
>	복수 column 추출 시, 리스트 활용
>
>   ```python
>   cust[[‘cust_class’,’age’,’r3m_avg_bill_amt’]] #[]안의 리스트가 가진 3개 컬럼만
>   ```

2.	원하는 row를 슬라이싱하기

>	하나의 값만 사용
>
>   ```python
>   cust[7:10] #정상
>   cust[7] #오류
>   cust[‘base_yn’] #정상
>   ```

데이터 조회의 2가지 방법

1.	loc

>	df의 인덱스번호, 컬럼명 기준으로 슬라이싱, 인덱싱 사용. 
>
>	df 행/열에 라벨로 접근 Location의 약자. 
>
>	인간이 읽을 수 있는 라벨값으로 데이터에 접근
>
>   ```python
>   cust.loc[102, ‘age’]
>   #cust데이터프레임에서 102번 인덱스의 ‘age’컬럼에 해당하는 값 추출.
>   ```

2.	iloc

>	내부적으로 붙여진 절대 인덱스 번호(0~n) 기준으로 행이나 컬럼 등에 접근
>
>	integer location의 약자. 컴퓨터가 읽을 수 있는 인덱싱값 이용.


***Boolean indexing**

데이터프레임에서 원하는 행들만 추출 시 사용

df2 데이터프레임에서 성별 컬럼에 남자만 추출
```python
df2[ df2[‘성별’]==’M’ ]
#성별 컬럼이 남자(M)인가?라는 질문에 참인 경우(Boolean)만을 골라 추출
```

dataFrame column 추가/삭제

추가

```python
cust[‘r3m_avg_bill_amt2’] = cust[‘r3m_avg_bill_amt’]*2
```

삭제
```python
cust.drop(‘r3m_avg_bill_amt3’,axis=1) #지정 컬럼 삭제
#axis=0은 행을 따라 내려가는 것, axis=1은 컬럼을 따라 쭉 오른쪽으로 가는 것.
```


### pandas 이해 및 활용 - DataFrame 변형하기 (이론+실습)


**Groupby 함수**

범주형 컬럼을 기준으로 같은 값을 묶어 통계 또는 집계 결과를 얻기 위해 사용하는 함수

> ex) 성별 컬럼의 값이 F인 데이터와, M인 데이터로 나누어 F그룹과 M그룹을 나누는 것.

내부적으로 데이터 분할(split) -> 적용(applying) -> 데이터 병합(combine) 의 3단계를 거쳐 수행된다.

>   ```python
>   dataframe.groupby(‘성별’).mean()
>   ```
>
>   #’성별’컬럼의 특징을 기준으로, 다른 컬럼(예를 들어, 나이)의 평균(mean)을 집계하여 데이터프레임으로 출력

분류 데이터의 특징을 파악할 때, 분류를 기준으로 groupby 함수를 사용해 연속형 컬럼의 평균이나 합계 등의 특징을 한눈에 보고 비교할 수 있어 자주 사용한다.


**pivot_table**

dataframe 형태를 변경하는 것.

인덱스에 표시될 데이터와 컬럼에 표시될 데이터가 둘 다 범주형 데이터여야 한다.
 
[img 2]

‘index’컬럼과 ‘columns’컬럼의 범주들을 각기 행, 열로 가지고 values를 값으로.

형식
> pandas.pivot_table(data, index, columns, aggfunc)
>
> pandas : 라이브러리 pandas
>
> data : 어떤 데이터프레임을 쓸 것인지. 데이터프레임의 이름.
>
> index : 인덱스 값으로 쓸 데이터의 컬럼 이름
>
> columns : 컬럼 값으로 쓸 데이터의 컬럼 이름
>
> aggfunc : 어떤 집계함수를 사용할 것인가.
>
> * 집계함수 : aggregate function. 데이터를 군집으로 묶어 요약된 통계 정보를 제공하는 함수. pandas에서는 주로 sum(합계), mean(평균), std(표준편차), max(최대값), min(최소값), median(중간값) 등이 있다.


**stack, unstack**

데이터 프레임을 재구조화하기 위해 사용하는 함수들

stack : 컬럼 레벨에서 보이는 컬럼명들을 인덱스 레벨로 가져와 데이터프레임 변경

unstack : 인덱스 레벨에서 보이는 인덱스값을 컬럼레벨로 보내어 dataframe 변경

* MultiIndex : 말 그대로, 여러 개의 인덱스를 사용함. set_index로도 지정이 가능하다.
```python
df1 = df.set_index('지점')
df1
df1.reset_index()
```

데이터프레임.reset_index() : 기존 인덱스 전부 컬럼으로 돌리고, 0~n의 기본 인덱스로 리셋.

### Pandas 이해 및 활용 – DataFrame 병합하기 (이론 + 실습)


**concat 함수**

동일 컬럼명을 갖는 데이터프레임끼리 합치는 경우 주로 사용.

동일한 형태의 데이터프레임을 단순히(보통 위/아래로) 합칠 때 사용

>ex) pandas.concat([A,B])

결과 데이터프레임으로 반환.

숨겨진 요소로 axis=0(디폴트)가 있다. 합칠 때, 좌-우로 합치려면 1로


**merge 함수**

두 데이터프레임을 공통된 컬럼을 기준으로 합치는 것.(공통 컬럼은 최소 1개로 충분!)

4개 방식 사용

>inner 방식	: 공통된 key 값으로 모은 교집합
>
>left 방식	: 왼쪽 데이터 프레임 기준으로 병합
>
>right 방식	: 오른쪽 데이터 프레임 기준으로 병합
>
>outer 방식	: 합집합으로 왼쪽-오른쪽 모드 합함.

>ex) pandas.merge(dfA, dfB, how=’inner’, on=’공통컬럼명’)
