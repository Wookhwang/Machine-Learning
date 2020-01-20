import AdalinGD as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 붓꽃 데이터를 주소를 통해 받아온다. Header=None를 해주어 attribute의 명칭을 기록하는 행을 삭제해준다.
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# setosa와 versicolor 선택
# setosa이면 -1을 할당해준다. 자동적으로 versicolor이면 1을 할당해준다.
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 꽃받침 길이와 꽃잎 길이를 추출합니다.
# pandas의 values 메서드를 사용해서 numpy array를 반환받는다.
X = df.iloc[0:100, [0, 2]].values

# 에포크 횟수 대비 비용 그래프 생성
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = ad.AdalinGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = ad.AdalinGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
