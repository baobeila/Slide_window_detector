"""pip install graphviz
pip install pydot
pip install pydot_ng
apt install graphviz
"""
import keras
import pydot as pyd
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from model import build_modle

keras.utils.vis_utils.pydot = pyd
from keras.utils import plot_model


# Visualize Model
def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(" nms")



# def visualize_model(model):
#     return SVG(model_to_dot(model).create(prog='dot', format='svg'))
if __name__ == '__main__':

    # create your model
    # then call the function on your model
    model = build_modle(input_shape=(64, 64, 3), classes=3)
    plot_model(model, to_file='model1.png')

