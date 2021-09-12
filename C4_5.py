import numpy as np
from ID3 import ID3Node
from sklearn.base import BaseEstimator
from copy import deepcopy


class C4_5Node(ID3Node):
    def __init__(
        self,
        split_attr=-1,
        value=None,
        depth=0,
        continuous=False,
        parent=None,
    ) -> None:
        super().__init__(split_attr=split_attr, value=value)
        self.depth = depth
        self.threshold: float = None
        self.continuous = continuous
        self.parent = parent
        self.train_list = None


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

    def fit(self, X, y):
        # 确定各列属性是连续还是离散
        self.continue_attr = [C4_5Classifier.is_number(x) for x in X[0]]
        self.X, self.y = np.array(X), np.array(y)
        # self.X的类型不再是数字，而是Object
        self.n_features = self.X.shape[1]
        self.leaf_list = []
        if self.min_samples_split < 1:
            self.min_samples_split *= len(self.X)
        if self.min_samples_leaf < 1:
            self.min_samples_leaf *= len(self.X)
        self.root = C4_5Node()
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
            if len(unique, ) == 1 or len(
                    attr_set,
            ) == 0 or len(np.unique(node_X[:, attr_set], axis=0)) == 1 or len(
                    id_list
            ) <= self.min_samples_split or node.depth >= self.max_depth:
                node.value = prior
                node.train_list = id_list
                self.leaf_list.append(node)
            else:
                node.split_attr, (_, _, node.threshold) = self.get_best_attr(
                    id_list, attr_set)
                attr_set.remove(node.split_attr)
                if self.continue_attr[node.split_attr]:  # 连续属性
                    X_attr = node_X[:, node.split_attr].astype(float)
                    index_left = X_attr <= node.threshold
                    index_right = np.logical_not(index_left)
                    node.continuous = True
                    left, right = id_list[index_left], id_list[index_right]

                    if min(len(left), len(right)) < self.min_samples_leaf:
                        node.value = prior
                        node.train_list = id_list
                        self.leaf_list.append(node)
                        continue

                    node.children = (
                        C4_5Node(depth=node.depth + 1, parent=node),
                        C4_5Node(depth=node.depth + 1, parent=node),
                    )

                    stack.append((node.children[1], right, attr_set.copy()))
                    stack.append((node.children[0], left, attr_set.copy()))
                else:  # 离散属性
                    unique = np.unique(self.X[:, node.split_attr])
                    for u in unique:
                        node.children[u] = C4_5Node(depth=node.depth + 1,
                                                    parent=node)
                        child_id_list = id_list[node_X[:,
                                                       node.split_attr] == u]
                        if len(child_id_list) == 0:  # 某特征在当前节点不存在
                            node.children[u].value = prior
                            self.leaf_list.append(node.children[u])
                        else:
                            stack.append((node.children[u], child_id_list,
                                          attr_set.copy()))

        return self

    def get_best_attr(self, id_list, attr_set):
        X, y = self.X[id_list], self.y[id_list]
        ent0 = C4_5Classifier.ent(y)

        attr_gain_dict = {}
        for attr in attr_set:
            if self.continue_attr[attr] is True:
                # 连续属性，计算出信息增益最大的分割阈值
                X_attr = X[:, attr].astype(float)
                x_set = np.unique(X_attr)
                if len(x_set) == 1:  # 无法分割
                    attr_gain_dict[attr] = (x_set[0], -np.inf)
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

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        stack = []
        stack.append((self.root, np.arange(len(X))))
        while len(stack) > 0:
            node, id_list = stack.pop()
            if node.value is None:
                data = X[id_list]
                if node.continuous:
                    index_left = data[:, node.split_attr].astype(
                        float) <= node.threshold
                    index_right = np.logical_not(index_left)
                    stack.append((
                        node.children[1],
                        id_list[index_right],
                    ))
                    stack.append((
                        node.children[0],
                        id_list[index_left],
                    ))
                else:
                    for attr_value, child in node.children.items():
                        stack.append((
                            child,
                            id_list[data[:, node.split_attr] == attr_value],
                        ))
            else:
                node.predict_list = id_list
        pred = np.zeros(len(X), dtype='int')
        for leaf in self.leaf_list:
            if leaf.predict_list is not None:
                pred[leaf.predict_list] = leaf.value
                leaf.predict_list = None
        return pred

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y)
        pred = self.predict(X)
        return np.mean(y == pred)

    # def pep_pruning(self) -> bool:
    #     candidate_parent = set()
    #     for leaf in self.leaf_list:
    #         parent = leaf.parent
    #         if parent.continuous and parent.children[
    #                 0].value != None and parent.children[1].value != None:
    #             candidate_parent.add(parent)
    #         elif parent.continuous == False:
    #             for child in parent.children.values():
    #                 if child.value == None:
    #                     continue
    #             candidate_parent.add(parent)

    #     candidate_parent = list(candidate_parent)
    #     print(len(candidate_parent))
    #     has_prune = False
    #     while len(candidate_parent) > 0:
    #         parent = candidate_parent.pop(0)  # 队列
    #         sub_n_leaves = len(parent.train_list)  # 子树的训练样本数
    #         error_sum = sum([
    #             np.sum(self.y[leaf.train_list] != leaf.value) + 0.5
    #             for leaf in parent.children
    #         ])  # 误差和
    #         error_ratio = error_sum / sub_n_leaves  # 误差率和
    #         error_std = np.sqrt(error_sum * (1 - error_ratio))
    #         pessimistic_error = error_sum + error_std  # 悲观误差值
    #         post_prune_error = 1 - self.__prune_test(parent, self.X, self.y)
    #         if post_prune_error * len(self.X) + 0.5 <= pessimistic_error:
    #             print("修正误差：{}; 悲观误差：{}".format(
    #                 post_prune_error,
    #                 pessimistic_error,
    #             ))
    #             unique, counts = np.unique(
    #                 self.y[parent.train_list],
    #                 return_counts=True,
    #             )
    #             parent.value = unique[np.argmax(counts)]
    #             self.leaf_list.append(parent)
    #             self.leaf_list.remove(parent.children[0])
    #             self.leaf_list.remove(parent.children[1])
    #             parent = parent.parent
    #             if parent.children[0].value != None and parent.children[
    #                     1].value != None:
    #                 candidate_parent.append(parent)

    #             has_prune = True

    #     return has_prune

    def rep_pruning(self, valid_X, valid_y) -> bool:
        # 一种低效的方法实现rep剪枝
        valid_X = np.array(valid_X).reshape(-1, self.n_features)
        valid_y = np.array(valid_y).reshape(-1)
        candidate_parent = set()
        for leaf in self.leaf_list:
            parent = leaf.parent
            if parent.continuous and parent.children[
                    0].value != None and parent.children[1].value != None:
                candidate_parent.add(parent)
            elif parent.continuous == False:
                for child in parent.children.values():
                    if child.value == None:
                        continue
                candidate_parent.add(parent)

        candidate_parent = list(candidate_parent)
        score = self.score(valid_X, valid_y)
        has_prune = False
        while len(candidate_parent) > 0:
            parent = candidate_parent.pop(0)  # 队列
            new_score = self.__prune_test(parent, valid_X, valid_y)
            if score <= new_score:
                score = new_score
                unique, counts = np.unique(
                    self.y[parent.train_list],
                    return_counts=True,
                )
                parent.value = unique[np.argmax(counts)]
                self.leaf_list.append(parent)
                self.leaf_list.remove(parent.children[0])
                self.leaf_list.remove(parent.children[1])
                parent = parent.parent
                if parent.children[0].value != None and parent.children[
                        1].value != None:
                    candidate_parent.append(parent)

                has_prune = True
        return has_prune

    def __prune_test(self, node, test_X, test_y):
        "计算剪枝后树在数据集上的得分"
        unique, counts = np.unique(self.y[node.train_list], return_counts=True)
        node.value = unique[np.argmax(counts)]

        leaf_list = self.leaf_list.copy()
        # 模拟剪枝
        self.leaf_list.remove(node.children[0])
        self.leaf_list.remove(node.children[1])
        self.leaf_list.append(node)
        score = self.score(test_X, test_y)

        # 恢复
        node.value, self.leaf_list = None, leaf_list
        return score

    def get_n_leaves(self):
        return len(self.leaf_list)

    def get_depth(self):
        return max([leaf.depth for leaf in self.leaf_list])

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
