from graphviz import Digraph
from os import remove
from CART import CARTClassifier


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
        if node.cls == -1:
            if type(model) == CARTClassifier:
                if feature_names is not None:
                    attr = "{}={}?".format(
                        feature_names[node.split_attr],
                        node.threshold,
                    )
                else:
                    attr = "x[{}]≤{}?".format(node.split_attr, node.threshold)
                g.node('node{}'.format(node_id), label=attr, fontname=font)
                g.node('node%d' % node_id, label=attr)
                left, right = node.children
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
                left, right = node.children
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
                cls = target_names[node.cls]
            else:
                cls = "class_{}".format(node.cls)
            g.node(
                'node{}'.format(node_id),
                label=cls,
                shape='box',
                fontname=font,
            )

    g.view(filename=filename)
    remove(filename)
