import base64
import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title = "Confusion matrix",
                          cmap = plt.cm.Blues, save_flg = False):
    #classes = [str(i) for i in range(10)]
    classes = ['normal','DE_B007','DE_B014','DE_B021','DE_IR007','DE_IR014','DE_IR021','DE_OR007','DE_OR014','DE_OR021']
    labels = range(10)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=32)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, fontsize=20)
    #plt.yticks(tick_marks, classes, fontsize=20)
    plt.xticks(tick_marks, classes, fontsize=9,rotation=270)
    plt.yticks(tick_marks, classes, fontsize=9)

    print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    img = io.BytesIO()

    if save_flg:
        # plt.savefig("./confusion_matrix.png")
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url


    #plt.show()


def plot_confusion_matrix_fed(y_true, y_pred, title = "Confusion matrix",
                          cmap = plt.cm.Blues, save_flg = False):
    #classes = [str(i) for i in range(10)]
    classes = ['normal','DE_B','DE_IR','DE_OR','FE_B','FE_IR','FE_OR']
    labels = range(7)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=32)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, fontsize=20)
    #plt.yticks(tick_marks, classes, fontsize=20)
    plt.xticks(tick_marks, classes, fontsize=9,rotation=270)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    img = io.BytesIO()

    if save_flg:
        # plt.savefig("./confusion_matrix.png")
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url