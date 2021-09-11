import numpy as np


class ID3Node:
    def __init__(self, split_attr=-1, value=None) -> None:
        self.split_attr: int = split_attr
        self.children: dict = {}
        self.value: int = value
        self.continuous = False
        self.predict_list = None


class ID3Classifier():
    '''
    ID3决策树是最朴素的决策树分类器，特点：
    - 无剪枝
    - 只支持离散属性
    - 采用信息增益准则
    '''
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y).reshape(-1).astype('int')
        self.leaf_list = []
        self.n_features = self.X.shape[1]
        self.root = ID3Node()
        stack = [(
            self.root,
            np.arange(len(self.X)),
            list(range(self.n_features)),
        )]
        while len(stack) > 0:
            node, id_list, attr_set = stack.pop()
            node_X, node_y = self.X[id_list], self.y[id_list]
            unique, counts = np.unique(node_y, return_counts=True)
            prior = unique[np.argmax(counts)]
            if len(unique) == 1 or len(np.unique(
                    node_X[:, attr_set], axis=0)) == 1 or len(attr_set) == 0:
                node.value = prior
                self.leaf_list.append(node)
            else:
                node.split_attr = self.get_best_attr(id_list, attr_set)
                copy_set = attr_set.copy()
                copy_set.remove(node.split_attr)
                # 注意这里要考虑特征attr的所有可能值
                unique = np.unique(self.X[:, node.split_attr])
                for u in unique:
                    node.children[u] = ID3Node()
                    child_id_list = id_list[node_X[:, node.split_attr] == u]
                    if len(child_id_list) == 0:  # 某特征在当前节点不存在
                        node.children[u].cls = prior
                        self.leaf_list.append(node.children[u])
                    else:
                        stack.append(
                            (node.children[u], child_id_list, copy_set))
        
        self.n_leaf = len(self.leaf_list)
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        stack = [(self.root, np.arange(len(X)))]
        while len(stack) > 0:
            node, id_list = stack.pop()
            if node.value is None:
                data = X[id_list]
                for attr_value, child in node.children.items():
                    stack.append((
                        child,
                        id_list[data[:, node.split_attr] == attr_value],
                    ))
            else:
                node.predict_list = id_list
        pred = np.zeros(len(X))
        for leaf in self.leaf_list:
            if leaf.predict_list is not None:
                pred[leaf.predict_list] = leaf.value
                leaf.predict_list = None
        return pred

    def get_best_attr(self, id_list, attr_set):
        X, y = self.X[id_list], self.y[id_list]
        entropy_dict = {}
        for attr in attr_set:
            unique, counts = np.unique(X[:, attr], return_counts=True)
            entropy_dict[attr] = -sum([
                counts[i] * ID3Classifier.ent(y[X[:, attr] == unique[i]])
                for i in range(len(unique))
            ]) / len(X)
        return max(entropy_dict.items(), key=lambda d: d[1])[0]

    @staticmethod
    def ent(y):
        counts = np.unique(
            y,
            return_counts=True,
        )[1].astype(float)
        dist = counts / np.sum(counts)
        return -np.sum(dist * np.log2(dist))
