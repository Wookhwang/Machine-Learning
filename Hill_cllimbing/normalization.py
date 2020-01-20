from epochs_graph import *
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경제를 그립니다.
    # x1은 꽃받침 길이, x2는 꽃잎 길이
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # meshgrid()는 축에 해당하는 1차원 배열을 전달받아 벡터 공간의 모든 좌표를 담은 행렬을 반환해준다.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # reval()를 사용해서 xx1, xx2를 훈련 데이터와 같은 개수의 열이 되도록 1차원 행렬을 만들어준다.
    # 이후에 predict()를 사용해서 각 포인트에 대응하는 클래스 레이블 Z를 예측
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Z를 xx1. xx2와 같은 차원의 그리드로 크기를 변경
    Z = Z.reshape(xx1.shape)
    # contourf()로 등고선 그려준다
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    # 산점도 시각화
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0],
                    y=X[y == c1, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=c1,
                    edgecolors='black')


X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = ad.AdalinGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [normalized]')
plt.ylabel('petal length [normalized]')
plt.legend(loc='upper left')
# tight_layout()을 안하면 그래프가 어정쩡하게 나옴
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
