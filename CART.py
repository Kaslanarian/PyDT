from functools import reduce
import numpy as np
from C4_5 import C4_5Node, C4_5Classifier


class CARTNode(C4_5Node):
    def __init__(
        self,
        split_attr=-1,
        value=None,
        depth=0,
        continuous=False,
        parent=None,
    ) -> None:
        super().__init__(split_attr=split_attr,
                         value=value,
                         depth=depth,
                         parent=parent,
                         continuous=continuous)
        self.Ct = None
        self.CT = None
        self.gt = None


class CARTClassifier(C4_5Classifier):
    def __init__(self,
                 max_depth=np.inf,
                 min_samples_split=2,
                 min_samples_leaf=1) -> None:
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf)

    def fit(self, X, y):
        # 确定各列属性是连续还是离散
        self.continue_attr = [CARTClassifier.is_number(x) for x in X[0]]
        self.X, self.y = np.array(X), np.array(y)

        self.leaves_list = []
        self.n_features = self.X.shape[1]

        if self.min_samples_split < 1:
            self.min_samples_split *= len(self.X)
        if self.min_samples_leaf < 1:
            self.min_samples_leaf *= len(self.X)

        self.root = CARTNode()
        self.root.sample_list = np.arange(len(self.X))
        stack = [(self.root, list(range(self.n_features)))]

        while len(stack) > 0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.sample_list], self.y[node.sample_list]
            unique, counts = np.unique(node_y, return_counts=True)
            prior = unique[np.argmax(counts)]
            if len(unique, ) == 1 or len(
                    attr_set,
            ) == 0 or len(np.unique(node_X[:, attr_set], axis=0)) == 1 or len(
                    node.sample_list
            ) <= self.min_samples_split or node.depth >= self.max_depth:
                node.value = prior
                node.CT = node.Ct = (node_y != prior).mean()
                self.leaves_list.append(node)
            else:
                node.split_attr, (_, node.threshold) = self.__get_best_attr(
                    node.sample_list,
                    attr_set,
                )
                attr_set.remove(node.split_attr)
                if self.continue_attr[node.split_attr]:  # 连续属性
                    node.continuous = True
                    X_attr = node_X[:, node.split_attr].astype(float)
                    index_left = X_attr <= node.threshold
                else:  # 离散属性
                    index_left = node_X[:, node.split_attr] == node.threshold
                index_right = np.logical_not(index_left)
                left, right = (
                    node.sample_list[index_left],
                    node.sample_list[index_right],
                )

                if min(len(left), len(right)) < self.min_samples_leaf:
                    node.value = prior
                    node.CT = node.Ct = (node_y != prior).mean()
                    self.leaves_list.append(node)
                    continue

                left_node = CARTNode(depth=node.depth + 1, parent=node)
                right_node = CARTNode(depth=node.depth + 1, parent=node)
                left_node.sample_list, right_node.sample_list = left, right
                node.children.update({True: left_node, False: right_node})

                stack.append((node.children[False], attr_set.copy()))
                stack.append((node.children[True], attr_set.copy()))

        self.n_leaf = len(self.leaves_list)
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
                else:
                    index_left = data[:, node.split_attr] == node.threshold
                index_right = np.logical_not(index_left)
                node.children[True].predict_list = node.predict_list[
                    index_left]
                node.children[False].predict_list = node.predict_list[
                    index_right]
                stack.append(node.children[False])
                stack.append(node.children[True])
                # node.predict_list = None  # 将非叶节点的predict_list还原为none

        pred = np.zeros(len(X))
        for leaf in self.leaves_list:
            if leaf.predict_list is not None:
                pred[leaf.predict_list] = leaf.value
        return pred

    def ccp_pruning(self, alpha=0.5):
        for leaf in self.leaves_list:
            leaf.leaves.append(self.leaves_list.index(leaf))

        frontier = self.calculate_frontier()
        leaf_list = self.leaves_list.copy()
        while len(frontier) > 0:
            # 自下往上计算gt
            parent: CARTNode = frontier.pop(0)
            error_num = 0  # 子树上的误分数
            for child in parent.children.values():
                error_num += child.CT * len(child.sample_list)
                parent.leaves.extend(child.leaves)
                leaf_list.remove(child)

            # CCP剪枝所需要的信息
            node_y = self.y[parent.sample_list]
            unique, counts = np.unique(node_y, return_counts=True)
            prior = unique[np.argmax(counts)]
            parent.Ct = (node_y != prior).mean()
            parent.CT = error_num / len(parent.sample_list)
            parent.gt = (parent.Ct - parent.CT) / (len(parent.leaves) - 1)

            leaf_list.append(parent)
            parent = parent.parent
            if parent is None:
                break
            elif reduce(
                    lambda x, y: x and y,
                [(child in leaf_list) for child in parent.children.values()],
            ):
                frontier.append(parent)

        frontier = self.calculate_frontier()
        while len(frontier) > 0:
            parent: CARTNode = frontier.pop(0)
            if parent.gt <= alpha:
                node_y = self.y[parent.sample_list]
                unique, counts = np.unique(node_y, return_counts=True)
                prior = unique[np.argmax(counts)]
                parent.value = prior
                for child in parent.children.values():
                    self.leaves_list.remove(child)
                self.leaves_list.append(parent)
                parent.children.clear()
                parent = parent.parent
                if parent is None:
                    break
                elif reduce(
                        lambda x, y: x and y,
                    [(child in leaf_list)
                     for child in parent.children.values()],
                ):
                    frontier.append(parent)

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y)
        pred = self.predict(X)
        return np.mean(y == pred)

    def __get_best_attr(self, id_list, attr_set):
        X, y = self.X[id_list], self.y[id_list]

        attr_gini_dict = {}
        for attr in attr_set:
            X_attr = X[:, attr]
            if self.continue_attr[attr] is True:
                X_attr = X_attr.astype(float)
                x_set = np.unique(X_attr)
                if len(x_set) == 1:  # 无法分割
                    attr_gini_dict[attr] = (x_set[0], -np.inf)
                    continue
                t_set = [(x_set[i] + x_set[i + 1]) / 2
                         for i in range(len(x_set) - 1)]
                gini_list = [
                    -len(y_left := y[X_attr <= t]) *
                    CARTClassifier.gini(y_left) -
                    len(y_right := y[X_attr > t]) *
                    CARTClassifier.gini(y_right) for t in t_set
                ]
                argmax = np.argmax(gini_list)
                threshold = t_set[argmax]
                # 连续属性下，返回(Gini指数, 划分阈值)
                attr_gini_dict[attr] = (
                    gini_list[argmax],
                    threshold,
                )
            else:
                unique = np.unique(X_attr)  # attr特征所有可能取值
                # 离散属性下，返回(基尼指数，候选特征值)
                gini_list = [
                    -len(y_ := y[X_attr != u]) * CARTClassifier.gini(y_) -
                    len(y_ := y[X_attr == u]) * CARTClassifier.gini(y_)
                    for u in unique
                ]
                argmax = np.argmax(gini_list)
                threshold = unique[argmax]
                attr_gini_dict[attr] = (
                    gini_list[argmax],
                    threshold,
                )
        return max(attr_gini_dict.items(), key=lambda item: item[1][0])

    @staticmethod
    def gini(y):
        counts = np.unique(
            y,
            return_counts=True,
        )[1].astype(float)
        dist = counts / np.sum(counts)
        return 1 - np.sum(dist**2)

    def calculate_frontier(self) -> set:
        return super().calculate_frontier()


class CARTRegressor(CARTClassifier):
    def __init__(
        self,
        max_depth=np.inf,
        min_samples_split=2,
        min_samples_leaf=1,
    ) -> None:
        '''
        只支持连续属性
        '''
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf)

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

        self.root = CARTNode()
        self.root.sample_list = np.arange(len(self.X))
        stack = [(self.root, list(range(self.n_features)))]

        while len(stack) > 0:
            node, attr_set = stack.pop()
            node_X, node_y = self.X[node.sample_list], self.y[node.sample_list]
            unique = np.unique(node_y)
            prior = np.mean(node_y)

            if len(unique, ) == 1 or len(
                    attr_set,
            ) == 0 or len(np.unique(node_X[:, attr_set], axis=0)) == 1 or len(
                    node.sample_list
            ) <= self.min_samples_split or node.depth >= self.max_depth:
                node.value = prior
                self.leaves_list.append(node)
            else:
                node.split_attr, (_, node.threshold) = self.__get_best_attr(
                    node.sample_list,
                    attr_set,
                )
                attr_set.remove(node.split_attr)
                if self.continue_attr[node.split_attr]:  # 连续属性
                    node.continuous = True
                    X_attr = node_X[:, node.split_attr].astype(float)
                    index_left = X_attr <= node.threshold
                else:  # 离散属性
                    index_left = node_X[:, node.split_attr] == node.threshold
                index_right = np.logical_not(index_left)
                left, right = (
                    node.sample_list[index_left],
                    node.sample_list[index_right],
                )

                if min(len(left), len(right)) < self.min_samples_leaf:
                    node.value = prior
                    self.leaves_list.append(node)
                    continue

                left_node = CARTNode(depth=node.depth + 1, parent=node)
                right_node = CARTNode(depth=node.depth + 1, parent=node)
                left_node.sample_list, right_node.sample_list = left, right
                node.children.update({True: left_node, False: right_node})

                stack.append((node.children[False], attr_set.copy()))
                stack.append((node.children[True], attr_set.copy()))

        self.depth = max([leaf.depth for leaf in self.leaves_list])
        self.n_leaf = len(self.leaves_list)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y)
        pred = self.predict(X)
        return -np.mean((y - pred)**2)

    def rep_pruning(self, valid_X, valid_y):
        valid_X = np.array(valid_X).reshape(-1, self.n_features)
        valid_y = np.array(valid_y).reshape(-1)
        pred_valid = self.predict(valid_X)

        frontier = set()
        for leaf in self.leaves_list:
            parent: CARTNode = leaf.parent
            if parent != None and reduce(
                    lambda x, y: x and y,
                [child.value != None for child in parent.children.values()]):
                frontier.add(parent)
        frontier = list(frontier)

        while len(frontier) > 0:
            parent = frontier.pop(0)
            # 如果剪枝，parent对应的值
            parent_value = np.mean(self.y[parent.sample_list])
            # 剪枝前的验证集上误差
            if len(parent.predict_list) == 0:  # 样本未经过该子树，跳过
                continue
            else:
                pre_error = np.mean(
                    (pred_valid - valid_y)[parent.predict_list]**2)
            # 剪枝后的验证集上误差
            post_error = np.mean(
                (valid_y[parent.predict_list] - parent_value)**2)
            if pre_error >= post_error:
                for child in parent.children.values():
                    self.leaves_list.remove(child)
                self.leaves_list.append(parent)
                parent.children.clear()
                parent.value = parent_value
                parent = parent.parent
                if parent == None:  # 根节点
                    break
                if reduce(
                        lambda x, y: x and y,
                    [
                        child.value != None
                        for child in parent.children.values()
                    ],
                ):
                    frontier.append(parent)

    def __get_best_attr(self, id_list, attr_set):
        X, y = self.X[id_list], self.y[id_list]

        attr_gini_dict = {}
        for attr in attr_set:
            X_attr = X[:, attr]
            if self.continue_attr[attr] is True:
                X_attr = X_attr.astype(float)
                x_set = np.unique(X_attr)
                if len(x_set) == 1:  # 无法分割
                    attr_gini_dict[attr] = (x_set[0], np.inf)
                    continue
                t_set = [(x_set[i] + x_set[i + 1]) / 2
                         for i in range(len(x_set) - 1)]
                gini_list = [
                    CARTRegressor.square_sum(y[X_attr <= t]) +
                    CARTRegressor.square_sum(y[X_attr > t]) for t in t_set
                ]
                argmin = np.argmin(gini_list)
                threshold = t_set[argmin]
                # 连续属性下，返回(误差平方和, 划分阈值)
                attr_gini_dict[attr] = (
                    gini_list[argmin],
                    threshold,
                )
            else:
                unique = np.unique(X_attr)  # attr特征所有可能取值
                # 离散属性下，返回(误差平凡和，候选特征值)
                gini_list = [
                    CARTRegressor.square_sum(y[X_attr == u]) +
                    CARTRegressor.square_sum(y[X_attr != u]) for t in t_set
                    for u in unique
                ]
                argmin = np.argmin(gini_list)
                threshold = unique[argmin]
                attr_gini_dict[attr] = (
                    gini_list[argmin],
                    threshold,
                )
        return min(attr_gini_dict.items(), key=lambda item: item[1][1])

    @staticmethod
    def square_sum(y):
        return np.sum((y - y.mean())**2)