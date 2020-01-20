import numpy as np


# gradient descent는 tensorflow에 이미 구현이 되어있다.
# 아달린은 가중치를 업데이트하는 데 단위 계산 함수(net_input)이 아니라 선형 활성화 함수를 사용한다.
class AdalinGD(object):
    """적응형 선형 뉴런 분류기

    매개변수
    ------------
    eta : float
      학습률 (0.0과 1.0 사이)
    n_iter : int
      훈련 데이터셋 반복 횟수
    random_state : int
      가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------
    w_ : 1d-array
      학습된 가중치
    cost_ : list
      에포크마다 누적된 비용 함수의 제곱합

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """훈련 데이터 학습

                매개변수
                ----------
                X : {array-like}, shape = [n_samples, n_features]
                  n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
                y : array-like, shape = [n_samples]
                  타깃값

                반환값
                -------
                self : object

                """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            # 특성 행렬과 오차 벡터 간의 행렬-벡터 곱셈
            self.w_[1:] += self.eta * X.T.dot(errors)
            # 절편(0번째 가중치)
            self.w_[0] += self.eta * errors.sum()
            # 비용을 cost_ 에 모아서 훈련이 수렴하는지 확인
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 단순 항등 함수 (단일층 시녕망을 통해 정보의 흐름을 표현 하기 위한 함수)
    def activation(self, X):
        """선형 활성화 계산"""
        return X

    def predict(self, X):
        """단위 계산 함수를 사용하여 클래스 테이블을 반환합니다."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    # 입력 데이터의 특성에서 최종 입력, 활성화, 출력 순으로 진행
