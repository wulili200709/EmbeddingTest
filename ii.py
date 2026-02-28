from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设有100个样本，每个1280维
features = [...所有样本的1280维特征...]
labels = [...对应的OK/NG标签...]

# 降维到2D
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(features)

# 画图
plt.scatter(features_2d[labels=='OK'], color='blue')
plt.scatter(features_2d[labels=='NG'], color='red')
plt.show()