import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import os
from MindsEye.clustering.equal_groups import EqualGroupsKMeans

feat_train_file = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/clip_feat.pickle' # train features
feat_test_file = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/clip_feat_test.pickle' # train features
outdir = '/fsx/proj-medarc/fmri/natural-scenes-dataset/clipfeat/'

with open(feat_train_file, 'rb') as handle:
    data_train = pickle.load(handle)

with open(feat_test_file, 'rb') as handle:
    data_test = pickle.load(handle)

X_train = []
files_train = []

X_test = []
files_test = []

for k in data_train.keys():
    # print(k, data[k][0], data[k].shape)
    X_train.append(data_train[k])
    files_train.append(k)

for k in data_test.keys():
    # print(k, data[k][0], data[k].shape)
    X_test.append(data_test[k])
    files_test.append(k)


X_train = np.stack(X_train, 0)  # 8559 x 512
X_test = np.stack(X_test, 0)  # 982 x 512
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

k = 10


kmeans = KMeans(n_clusters=k, max_iter=1000, tol=1e-7).fit(X_train)
# kmeans = EqualGroupsKMeans(n_clusters=k ).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_ # k x 512

centers = preprocessing.normalize(centers)


cosine = X_test @ centers.T
labels_test = np.argmax(cosine, 1)


count = {}
# count labels
for l in labels:
    if l in count.keys():
        count[l] += 1
    else:
        count[l] = 1

co = []
for c in range(k):
    print(c, count[c])
    co.append(count[c])


plt.bar(np.arange(k), co)
plt.title('Population of classes')
plt.xlabel('class')
plt.ylabel('Population')
plt.savefig('/workspace/mycode/MindsEye/output/class_pop.png')


plt.show()


pseudo_lbls_train = {}
for i in range(len(X_train)):
    pseudo_lbls_train[files_train[i]] = labels[i]

pseudo_lbls_test = {}
for i in range(len(X_test)):
    pseudo_lbls_test[files_test[i]] = labels_test[i]

with open(os.path.join(outdir, 'pseudo_lbls_train.pickle'), 'wb') as handle:
    pickle.dump(pseudo_lbls_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(outdir, 'pseudo_lbls_val.pickle'), 'wb') as handle:
    pickle.dump(pseudo_lbls_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(labels.shape)
print(labels.min(), labels.max())

pca = PCA(n_components=2)

# X_2d = pca.fit_transform(X)
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_train)
# print(X_embedded.shape)
# exit()

print(X_2d.shape)

plt.title('tSNE visualization of the clusters')
colors = cm.rainbow(np.linspace(0, 1, k))
for i in range(k):
    mask = labels == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=colors[i])

plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.savefig('/workspace/mycode/MindsEye/output/clustering.png')
plt.show()
