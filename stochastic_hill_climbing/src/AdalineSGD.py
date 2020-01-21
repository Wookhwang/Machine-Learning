import numpy as np


# gradient descent는 tensorflow에 이미 구현이 되어있다.
# 확률적 선형 뉴런 분석기는 각 훈련 샘플에 대해서 조금씩 가중치를 업데이트 한다.
class AdalineSGD(object):
    """Adaptive Linear Neuron 분류기

    매개변수
    ------------
    eta : float
      학습률 (0.0과 1.0 사이)
    n_iter : int
      훈련 데이터셋 반복 횟수
    shuffle : bool (default true)
        True로 설정하면 같은 반복이 되지 않도록 에포크마다 훈련 데이터를 섞습니다.
    random_state : int
      가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------
    w_ : 1d-array
      학습된 가중치
    cost_ : list
      에포크마다 누적된 비용 함수의 제곱합

    """
    def __init__(self, eta=0.01, n_iter=10, suffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.suffle = suffle
        self.random_state = random_state
        self.w_initialize = False

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
        # 언제나 그렇듯 fit()은 분류기 훈련용
        # 대신에 확률적 분류기에서는 _initialized_weights()을 사용해 행 갯수 만큼 가중치를 초기화
        self._initialized_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # _suffle()을 사용해서 훈련 데이터를 섞어줌. True일 때만
            if self.suffle:
                X, y = self._suffle(X, y)
            cost = []
            # 가중치 업데이트 하고, cost도 update해준다.
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            # 평균 비용을 구해서 cost_ 리스트에 추가시켜준다.
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    # partial_fit()은 온라인 학습용으로 구현 함.
    # 지속적으로 업데이트 되는 훈련 셋이 들어올 때 사용
    def partial_fit(self, X, y):
        """가중치를 다시 초기화하지 않고 훈련 데이터를 학습합니다."""
        if not self.w_initialize:
            self._intialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _suffle(self, X, y):
        """훈련 데이터를 섞음"""
        # permatation()을 통해 0~100까지 중복되지 않은 랜덤한 숫자 시퀸스를 생성 (y의 길이만큼)
        # 이 숫자 시퀸스는 특성 행렬과 클래스 레이블 백터를 섞는 인덱스로 활용
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialized_weights(self, m):
        """랜덤한 작은 수로 가중치를 초기화"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialize = True

    def _update_weights(self, xi, target):
        """아달린 학습 규칙을 적용하여 가중치를 업데이트"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

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

