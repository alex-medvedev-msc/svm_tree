from svm_tree import SVMTree
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np

tree = SVMTree()

print(tree.grid_search(10))

best = (0.97394736842105267, 32768, 0.125)