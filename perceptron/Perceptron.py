import numpy as np


class Perceptron(object):
    """퍼셉트론 분류기

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
    errors_ : list
      에포크마다 누적된 분류 오류

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
        # n_feature + 1만큼 벡터공간을 초기화
        # 1을 더한 이유는 벡터의 첫번쨔 원소인 절편을 위해서이다.
        # 이후에 가중치를 RandomState()를 사용해서 표준편차가 0.01인 정규 분포에서 뽑은 랜덤한 작은 수로 초기화
        # rgen은 numpy 난수 생성기이다.
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # zip() 테스트 코드 xi, target에 zip(X, y)의 값을 각각 넣어주고,
                # target은 진짜 클레스 레이블 값으로 사용, xi는 예측 레이블 값으로 사용한다.
                print(xi, target) # test
                # 가장 중요한 가중치 업데이트 부분
                # 레이블 값을 predict()를 이용해서 확인한뒤 가중치를 업데이트 해준다.
                # 이때 한꺼번에 일괄적으로 업데이트를 진행한다.
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 예측이 성공적이면 업데이트가 일어나지 않는다.
                # 때문에 업데이트 시에 에러를 기록해주기 위해 error에 update값을 정수형으로 저장해주고,
                # 기존에 만들어 놓은 errors_ 리스트에 추가시킨다.
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        # dot()를 통해 W(transport)*X를 계산한다. 그리고 가중치의 절편을 더해준다.
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        # dot()를 통해 W(transport)*X를 계산한다. 그리고 가중치의 절편을 더해준다.
        return np.where(self.net_input(X) >= 0.0, 1, -1)
