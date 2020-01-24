import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
"""
np.unique()는 target에 저장된 세 개의 고유한 클래스 레이블을 저장
여기서 iris는 모두 정수 레이블로 저장되어 있다.
정수 레이블로 저장해야하는 이유는 작은 메모리 영역을 차지해 계산 성능을 향상시킬 수 있기 때문이다.
즉, 대부분의 머신러닝 라이브러리에서는 레이블들이 정수화 되어있다.
"""

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target;
print('클래스 레이블: ', np.unique(y))


"""
model_selection 모듈의 train_test_split 함수를 사용해서 X와 y의 배열을 랜덤하게 나눈다.
test_size 값을 0.3으로 설정해서 30%는 테스트 데이터, 70%는 훈련 데이터가 된다.
분할 전 random_state 값을 설정해 데이터셋을 미리 섞는다.
stratify는 계층화에 사용된다. 계층화란 훈련 세트, 테스트 세트의 비율을 입력 데이터 세트의 비율과 같게 만드는 것이다.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# 계층화 확인 코드
print('y의 레이블 카운트 :', np.bincount(y))
print('y_train의 레이블 카운트 :', np.bincount(y_train))
print('y_test의 레이블 카운트 :', np.bincount(y_test))


"""
StandardScaler를 통해서 특성을 표준화
sc에 StandardScaler객체를 할당해 준다. 
StandardScaler의 fit 메소드는 훈련 세트의 각 특성 차원 마다 샘플 평균과 표준 편차를 계산 
transform 메소드는 샘플 평균과 표준 편차를 사용해 훈련세트를 표준화한다.
이후에 퍼셉트론 모델을 훈련한다.
OvR(One-versus-Rest) 방식으로 다중 분류
"""
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


"""
적절한 학습률을 찾기 위해서 linear_model 모듈의 Perceptron을 사용한다.
"""
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
ppn.fit(X_train_std, y_train)


"""
분류오차 = 1/45 := 0.022 := 2.2%
정확도 = 1 - 분류오차 = 97.8%
"""
y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())


"""
테스트 세트에서의 퍼셉트론 분류 정확도 계산
y_test는 실제 클래스 레이블, y_pred는 예측 클래스 레이블이다.
"""
print('정확도: %.2f' % accuracy_score(y_test, y_pred))


"""
테스트 세트에서의 퍼셉트론 분류 정확도 계산
X_test_std는 표준화된 실제 클래스 레이블, y_test는 실제 클래스 레이블이다.
"""
print('정확도: %.2f' % ppn.score(X_test_std, y_test))
