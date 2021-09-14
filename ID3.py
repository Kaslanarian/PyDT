import numpy as np
from functools import reduce


class ID3Node:
    def __init__(self, split_attr=-1, value=None, parent=None) -> None:
        # 分割属性
        self.split_attr: int = split_attr
        # 子节点
        self.children: dict = {}
        # 父节点
        self.parent = parent
        # 叶节点的值
        self.value: int = value
        # 预测列表
        self.predict_list: list = None
        # 落到该节点训练样本列表
        self.sample_list: list = None
        # 属性连续性 ID3中必然是False
        self.continuous = False


class ID3Classifier():
    '''
    ID3决策树是最朴素的决策树分类器，特点：
    - 无剪枝，但我们在这里加入REP剪枝
    - 只支持离散属性
    - 采用信息增益准则
    '''
    def __init__(self) -> None:
        super().__init__()
        self.n_leaves = 0

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y).reshape(-1)
        self.leaves_list = []
        self.n_features = self.X.shape[1]
        self.root = ID3Node()
        self.root.sample_list = np.arange(len(self.X))

        stack = [(self.root, list(range(self.n_features)))]
        while len(stack) > 0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.sample_list], self.y[node.sample_list]
            unique, counts = np.unique(node_y, return_counts=True)
            prior = unique[np.argmax(counts)]
            # 落入该节点样本类别相同 or 样本属性值相同 or 属性集为空
            if len(unique) == 1 or len(np.unique(
                    node_X[:, attr_set],
                    axis=0,
            )) == 1 or len(attr_set) == 0:
                node.value = prior
                self.leaves_list.append(node)
            else:
                node.split_attr = self.__get_best_attr(
                    node.sample_list,
                    attr_set,
                )
                copy_set = attr_set.copy()
                copy_set.remove(node.split_attr)
                # 注意这里要考虑特征attr的所有可能值
                unique = np.unique(self.X[:, node.split_attr])
                for u in unique:
                    node.children[u] = ID3Node(parent=node)
                    child_id_list = node.sample_list[node_X[:, node.split_attr]
                                                     == u]
                    node.children[u].sample_list = child_id_list
                    if len(child_id_list) == 0:  # 某特征在当前节点不存在
                        node.children[u].value = prior
                        self.leaves_list.append(node.children[u])
                    else:
                        stack.append((node.children[u], copy_set))

        self.n_leaves = len(self.leaves_list)
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        for leaf in self.leaves_list:
            leaf.predict_list = None
        self.root.predict_list = np.arange(len(X))
        stack = [self.root]

        while len(stack) > 0:
            node = stack.pop()
            if node.value is None:
                data = X[node.predict_list]
                for attr_value, child in node.children.items():
                    child.predict_list = node.predict_list[
                        data[:, node.split_attr] == attr_value]
                    stack.append(child)

        pred = np.zeros(len(X))
        for leaf in self.leaves_list:
            if leaf.predict_list is not None:
                pred[leaf.predict_list] = leaf.value
        return pred

    def rep_pruning(self, valid_X, valid_y):
        '''
        用验证数据valid_X, valid_y进行剪枝
        '''
        valid_X = np.array(valid_X).reshape(-1, self.n_features)
        valid_y = np.array(valid_y).reshape(-1)
        pred_valid = self.predict(valid_X)

        frontier = set()
        for leaf in self.leaves_list:
            parent: ID3Node = leaf.parent
            if parent != None and reduce(
                    lambda x, y: x and y,
                [child.value != None for child in parent.children.values()]):
                frontier.add(parent)
        frontier = list(frontier)
        while len(frontier) > 0:
            parent = frontier.pop(0)
            # 如果剪枝，parent对应的标签
            unique, counts = np.unique(self.y[parent.sample_list],
                                       return_counts=True)
            parent_cls = unique[np.argmax(counts)]

            # 测验证集在parent下不剪枝和不剪枝的准确率
            n_samples, n_unprune_right, n_prune_right = 0, 0, 0
            for child in parent.children.values():
                if child.predict_list != None:
                    n_samples += len(child.predict_list)
                    n_unprune_right += sum(pred_valid[child.predict_list] ==
                                           valid_y[child.predict_list])
                    n_prune_right += sum(
                        parent_cls == valid_y[child.predict_list])
            if n_samples == 0:  # 验证集不经过parent,无法剪枝，跳过
                continue
            # 比较
            if n_prune_right >= n_unprune_right:
                # step1 : leaves_list
                for child in parent.children.values():
                    self.leaves_list.remove(child)
                self.leaves_list.append(parent)
                # step2 : parent信息更新
                parent.children.clear()
                parent.value = parent_cls
                # step3 : 更新frontier
                parent = parent.parent
                if parent == None:  # 根节点
                    break
                if reduce(
                        lambda x, y: x and y,
                    [child.value != None for child in parent.children.values()],
                ):
                    frontier.append(parent)

    def __get_best_attr(self, id_list, attr_set):
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