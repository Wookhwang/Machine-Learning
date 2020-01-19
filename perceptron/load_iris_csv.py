import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Perceptron as pr

# 붓꽃 데이터를 주소를 통해 받아온다. Header=None를 해주어 attribute의 명칭을 기록하는 행을 삭제해준다.
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# setosa와 versicolor 선택
# setosa이면 -1을 할당해준다. 자동적으로 versicolor이면 1을 할당해준다.
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 꽃받침 길이와 꽃잎 길이를 추출합니다.
# pandas의 values 메서드를 사용해서 numpy array를 반환받는다.
X = df.iloc[0:100, [0, 2]].values


# 산점도를 그립니다.
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# 에포크 대비 잘못 부뉴된 오차를 그래프로 보여 줍니다.
# 에포크 : 훈련 데이터 셋을 반복할 최대 횟수
# plot()에서 error의 갯수를 알기 위해 len()을 사용해 errors_의 길이를 구한다.
ppn = pr.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

plt.show()
