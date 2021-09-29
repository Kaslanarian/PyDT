from graphviz import Digraph
from os import remove
from CART import CARTClassifier, CARTRegressor


def tree_plot(
    model,
    filename="tree",
    feature_names: list = None,
    target_names: list = None,
    font=None,
):
    '''
    利用graphviz模块绘制决策树
    '''
    if "root" not in model.__dir__():
        print("model is not fitted")
        return -1
    g = Digraph(format='png')
    i = 0
    stack = [(model.root, i)]
    while len(stack) > 0:
        node, node_id = stack.pop()
        if node.value == None:
            if type(model) in {CARTClassifier, CARTRegressor}:
                if feature_names is not None:
                    attr = "{}{}{}?".format(
                        feature_names[node.split_attr],
                        "≤" if node.continuous else "=",
                        node.threshold,
                    )
                else:
                    attr = "x[{}]{}{}?".format(
                        node.split_attr,
                        "≤" if node.continuous else "=",
                        node.threshold,
                    )
                g.node('node{}'.format(node_id), label=attr, fontname=font)
                g.node('node%d' % node_id, label=attr)
                left, right = node.children.values()
                g.node('node%d' % (i + 1))
                g.node('node%d' % (i + 2))
                g.edge('node%d' % node_id, 'node%d' % (i + 1), label="True")
                g.edge('node%d' % node_id, 'node%d' % (i + 2), label="False")
                stack.append((right, (i + 2)))
                stack.append((left, (i + 1)))
                i += 2
            elif node.continuous:  # 连续节点
                if feature_names is not None:
                    attr = "{}≤{}?".format(
                        feature_names[node.split_attr],
                        node.threshold,
                    )
                else:
                    attr = "x[{}]≤{}?".format(node.split_attr, node.threshold)
                g.node('node{}'.format(node_id), label=attr, fontname=font)
                g.node('node%d' % node_id, label=attr)
                left, right = node.children.values()
                g.node('node%d' % (i + 1))
                g.node('node%d' % (i + 2))
                g.edge('node%d' % node_id, 'node%d' % (i + 1), label="True")
                g.edge('node%d' % node_id, 'node%d' % (i + 2), label="False")
                stack.append((right, (i + 2)))
                stack.append((left, (i + 1)))
                i += 2
            else:
                if feature_names is not None:
                    attr = feature_names[node.split_attr]
                else:
                    attr = "x[{}]".format(node.split_attr)
                g.node('node{}'.format(node_id), label=attr, fontname=font)
                for attr, child in node.children.items():
                    i += 1
                    g.node('node{}'.format(i))
                    g.edge('node{}'.format(node_id),
                           'node{}'.format(i),
                           label=attr,
                           fontname=font)
                    stack.append((child, i))
        else:
            if target_names is not None:
                value = target_names[node.value]
            else:
                value = "{}{}".format(
                    "" if type(model) == CARTRegressor else "class_",
                    node.value)
            g.node(
                'node{}'.format(node_id),
                label=value,
                shape='box',
                fontname=font,
            )

    g.view(filename=filename)
    remove(filename)
    return 0
