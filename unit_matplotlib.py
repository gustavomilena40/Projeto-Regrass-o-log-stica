import pandas as pd
import sklearn as sk 
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import itertools





def plot_grafico_X_Y(x,y,label_x,label_y):
    jet=plt.get_cmap('coolwarm')
    plt.scatter(x,y,s=100, c=y,cmap=jet,vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    return plt





def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
 
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Numero de exemplos de treino")
    axes[0].set_ylabel("Pontuação")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Pontuação do treino")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Pontuação da validação")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Numero de exemplos de treino")
    axes[1].set_ylabel("Numero de Fits")
    axes[1].set_title("Escalabilidade do modelo")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Numero de Fits")
    axes[2].set_ylabel("Pontuação")
    axes[2].set_title("Performance do modelo")

    return plt


def plot_confusion_matrix(cm, target_names, title='Matriz de confusão', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="lawngreen" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "red")
    plt.tight_layout()
    plt.ylabel('NMCs Corretas')
    plt.xlabel('NMCs Previstas\nacuracia={:0.4f}; erros={:0.4f}'.format(accuracy, misclass))
    plt.show()


#fig, axes = plt.subplots(3, 2, figsize=(10, 15))
'''
X, y = load_digits(return_X_y=True)
'''
'''
title = "Learning Curves (Naive Bayes)"
'''
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
'''
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
'''

#estimator = LogisticRegression()
'''
plot_learning_curve(logreg, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
'''
'''
title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
'''
# SVC is more expensive so we do a lower number of CV iterations:
'''
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
'''
#logreg = SVC(gamma=0.001)
'''
plot_learning_curve(logreg, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
'''
'''
plt.show()
'''