import numpy as np
from ID3 import ID3Node
from sklearn.base import BaseEstimator
from copy import deepcopy
from functools import reduce


class C4_5Node(ID3Node):
    def __init__(
        self,
        split_attr=-1,
        value=None,
        parent=None,
        depth=0,
        continuous=False,
    ) -> None:
        super().__init__(split_attr=split_attr, value=value, parent=parent)
        # 节点深度
        self.depth = depth
        # 连续属性下划分阈值
        self.threshold: float = None
        # 属性连续性
        self.continuous = continuous
        # 训练时的误分数
        self.error_num = -1
        # 训练时该节点拥有的叶子
        self.leaves: list = []


class C4_5Classifier(BaseEstimator):
    def __init__(
        self,
        max_depth=np.inf,
        min_samples_split=2,
        min_samples_leaf=1,
    ) -> None:
        '''
        Parameters
        ----------
        `max_depth`: 决策树最大深度，默认是不设限的，也就是inf
        `min_samples_split`: 生成节点的样本数下限，低于该下限则不再进行划分，整数则视为样本数，小数则视为占训练样本百分比
        `max_samples_leaf`: 叶节点样本数下限，整数则视为样本数，小数则视为占训练样本百分比
        '''
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_leaves = 0
        self.depth = 0

    def fit(self, X, y):
        # 确定各列属性是连续还是离散
        self.continue_attr = [C4_5Classifier.is_number(x) for x in X[0]]
        self.X, self.y = np.array(X), np.array(y)
        # self.X的类型不再是数字，而是Object

        self.leaves_list = []
        self.n_features = self.X.shape[1]

        if self.min_samples_split < 1:
            self.min_samples_split *= len(self.X)
        if self.min_samples_leaf < 1:
            self.min_samples_leaf *= len(self.X)

        self.root = C4_5Node()
        self.root.sample_list = np.arange(len(self.X))

        stack = [(self.root, list(range(self.n_features)))]
        while len(stack) > 0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.sample_list], self.y[node.sample_list]
            unique, counts = np.unique(node_y, return_counts=True)
            prior = unique[np.argmax(counts)]

            # 落入该节点样本类别相同 or 样本属性值相同 or 属性集为空
            if len(unique, ) == 1 or len(
                    attr_set,
            ) == 0 or len(np.unique(node_X[:, attr_set], axis=0)) == 1 or len(
                    node.sample_list
            ) <= self.min_samples_split or node.depth >= self.max_depth:
                node.value = prior
                node.error_num = sum(self.y[node.sample_list] != node.value)
                self.leaves_list.append(node)
            else:
                node.split_attr, (_, _, node.threshold) = self.__get_best_attr(
                    node.sample_list,
                    attr_set,
                )
                attr_set.remove(node.split_attr)
                if self.continue_attr[node.split_attr]:  # 连续属性
                    X_attr = node_X[:, node.split_attr].astype(float)
                    index_left = X_attr <= node.threshold
                    index_right = np.logical_not(index_left)
                    node.continuous = True
                    left, right = (
                        node.sample_list[index_left],
                        node.sample_list[index_right],
                    )

                    if min(len(left), len(right)) < self.min_samples_leaf:
                        # 样本过少不分裂
                        node.value = prior
                        node.error_num = sum(
                            self.y[node.sample_list] != node.value)
                        self.leaves_list.append(node)
                        continue

                    left_node = C4_5Node(depth=node.depth + 1, parent=node)
                    right_node = C4_5Node(depth=node.depth + 1, parent=node)
                    left_node.sample_list, right_node.sample_list = left, right
                    node.children.update({True: left_node, False: right_node})

                    stack.append((right_node, attr_set.copy()))
                    stack.append((left_node, attr_set.copy()))
                else:  # 离散属性
                    unique = np.unique(self.X[:, node.split_attr])
                    for u in unique:
                        node.children[u] = C4_5Node(
                            depth=node.depth + 1,
                            parent=node,
                        )
                        child_id_list = node.sample_list[
                            node_X[:, node.split_attr] == u]
                        node.children[u].sample_list = child_id_list
                        if len(child_id_list) == 0:  # 某特征在当前节点不存在
                            node.children[u].value = prior
                            node.children[u].error_num = 0
                            self.leaves_list.append(node.children[u])
                        else:
                            stack.append((node.children[u], attr_set.copy()))

        self.n_leaves = len(self.leaves_list)
        self.depth = max([leaf.depth for leaf in self.leaves_list])
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
                if node.continuous:
                    index_left = data[:, node.split_attr].astype(
                        float) <= node.threshold
                    index_right = np.logical_not(index_left)
                    node.children[True].predict_list = node.predict_list[
                        index_left]
                    node.children[False].predict_list = node.predict_list[
                        index_right]
                    stack.append(node.children[False])
                    stack.append(node.children[True])
                else:
                    for attr_value, child in node.children.items():
                        child.predict_list = node.predict_list[
                            data[:, node.split_attr] == attr_value]
                        stack.append(child)
                # node.predict_list = None  # 将非叶节点的predict_list还原为none

        pred = np.zeros(len(X))
        for leaf in self.leaves_list:
            if leaf.predict_list is not None:
                pred[leaf.predict_list] = leaf.value
        return pred

    def pep_pruning(self):
        # 首先要算所有非叶节点对应的子树拥有的叶节点
        for leaf in self.leaves_list:
            leaf.leaves.append(self.leaves_list.index(leaf))

        frontier = self.calculate_frontier()
        leaf_list = self.leaves_list.copy()
        while len(frontier) > 0:
            parent: C4_5Node = frontier.pop(0)
            # parent.leaves是子节点leaves的并
            for child in parent.children.values():
                parent.leaves.extend(child.leaves)
                leaf_list.remove(child)

            leaf_list.append(parent)
            parent = parent.parent
            if parent is None:
                break
            elif reduce(
                    lambda x, y: x and y,
                [(child in leaf_list) for child in parent.children.values()],
            ):
                # 如果该节点仍属于frontier
                frontier.append(parent)
        # 从上往下悲观剪枝
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            if node.value != None:  # 子节点
                continue
            # 计算误差和
            error_sum = sum(
                [self.leaves_list[i].error_num + 0.5 for i in node.leaves])
            n_leaves = sum(
                [len(self.leaves_list[i].sample_list) for i in node.leaves])
            # 误差率和
            error_ratio = error_sum / n_leaves
            error_std = np.sqrt(error_sum * (1 - error_ratio))
            pessimistic_error = error_sum + error_std  # 悲观误差值
            # 计算剪枝后的误分率
            unique, counts = np.unique(
                self.y[node.sample_list],
                return_counts=True,
            )
            prior = unique[np.argmax(counts)]  # 剪枝后该节点的标签
            prune_error_num = np.sum([self.y[node.sample_list] != prior
                                      ])  # 剪枝后的误分数
            if prune_error_num + 0.5 < pessimistic_error:
                # print("修正误差：{}; 悲观误差：{}".format(
                #     prune_error_num + 0.5,
                #     pessimistic_error,
                # ))
                node.value = prior
                self.leaves_list.append(node)
                temp_list = self.leaves_list.copy()
                for leaf_id in node.leaves:
                    self.leaves_list.remove(temp_list[leaf_id])
            else:
                stack.extend(node.children.values())

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y)
        pred = self.predict(X)
        return np.mean(y == pred)

    def get_n_leaves(self):
        return len(self.leaves_list)

    def get_depth(self):
        return max([leaf.depth for leaf in self.leaves_list])

    @staticmethod
    def ent(y):
        counts = np.unique(
            y,
            return_counts=True,
        )[1].astype(float)
        dist = counts / np.sum(counts)
        return -np.sum(dist * np.log2(dist))

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def __get_best_attr(self, id_list, attr_set):
        X, y = self.X[id_list], self.y[id_list]
        ent0 = C4_5Classifier.ent(y)

        attr_gain_dict = {}
        for attr in attr_set:
            if self.continue_attr[attr] is True:
                # 连续属性，计算出信息增益最大的分割阈值
                X_attr = X[:, attr].astype(float)
                x_set = np.unique(X_attr)
                if len(x_set) == 1:  # 无法分割
                    attr_gain_dict[attr] = (x_set[0], -np.inf, x_set[0])
                    continue
                t_set = [(x_set[i] + x_set[i + 1]) / 2
                         for i in range(len(x_set) - 1)]
                ent_list = [
                    -len(y_left := y[X_attr <= t]) * C4_5Classifier.ent(y_left)
                    -
                    len(y_right := y[X_attr > t]) * C4_5Classifier.ent(y_right)
                    for t in t_set
                ]
                argmax = np.argmax(ent_list)
                shreshold = t_set[argmax]
                # 连续属性下，返回(信息增益, 信息增益率, 划分阈值)
                gain = ent0 + ent_list[argmax] / len(id_list)
                attr_gain_dict[attr] = (
                    gain,
                    gain / C4_5Classifier.ent(X_attr <= shreshold),
                    shreshold,
                )
            else:
                # 离散属性
                X_attr = X[:, attr]
                unique, counts = np.unique(X_attr, return_counts=True)
                gain = ent0 - sum([
                    counts[i] * C4_5Classifier.ent(y[X_attr == unique[i]])
                    for i in range(len(unique))
                ]) / len(X)
                # 离散属性下，返回(信息增益, 信息增益率, None)
                attr_gain_dict[attr] = (
                    gain,
                    gain / C4_5Classifier.ent(X_attr),
                    None,
                )

        # 启发式方法：选择信息增益高于平均值的属性，再选择增益率最高的
        avg = np.mean([item[0] for item in attr_gain_dict.values()])
        # 按信息增益率排序, 再返回最大的属性
        for item in sorted(
                attr_gain_dict.items(),
                key=lambda d: d[1][1],
                reverse=True,
        ):
            if item[1][0] >= avg:
                break
        return item

    def calculate_frontier(self) -> set:
        candidate_parent = set()
        for leaf in self.leaves_list:
            parent = leaf.parent
            is_frontier = True
            for child in parent.children.values():
                if child.value == None:
                    is_frontier = False
                    break
            if is_frontier:
                candidate_parent.add(parent)
        return list(candidate_parent)