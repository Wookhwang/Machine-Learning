import matplotlib.pyplot as plt
import numpy as np

"""
sigmoid 함수
    - odds ratio : 특정 이벤트가 발생할 확률
        = P / (1 - P) 로 표현이 가능
    
    - logit(P) = log(P / (1 - P))
        -> logit(P(y=1 | x)) = (w**t) * x 
        -> P(y=1 | x)는 특성 x가 주어졌을 때 이 샘플이 클래스 1에 속할 조건부 확률
    
    - 즉 1 / logit(P)이 logistic sigmoid function이다.
    -> phi(z) = 1 / (1 + (e**(-z)))
    -> z는 가중치와 특성의 선형 조합으로 이루어진 최종입력
    -> z = (w**t) * x 
"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

"""
조건부 확률이기 때문에 클래스에 소속될 확률도 구할 수 있다.
그래프에서는 -inf로 갈때는 0에 수렴하고, inf로 갈때는 1에 수렴
"""
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y축의 눈금과 격자선
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

""" 
경사 하강법을 사용한 비용함수
y = 1일 때와 y = 0일 때의 함수를 구현
이후 그래프를 출력해 보면 예측의 정확도가 낮아질 수록 비용이 증가함을 알 수 있다.
"""
def cost_1(z):
    return -np.log(sigmoid(z))


def cost_0(z):
    return -np.log(1 - sigmoid(z))


z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()