import numpy as np
import pandas as pd
import scipy.spatial as sp


class ClasskNN:
    def __init__(self, k, lista_uczaca):
        self.k = k
        self.lista_uczaca = lista_uczaca

    def predict(self, lista_z_obiektami):
        return ""

    def score(self, lista_z_obiektami, lista_etykiet):
        return ""


listaUczaca = np.array(pd.read_csv("data-learning.csv", header=None))
listaTestujaca = np.array(pd.read_csv("data-test.csv", header=None))

listaUczacaBezEtykiet = np.delete(listaUczaca, len(listaUczaca[0])-1, axis=1)
listaTestujacaBezEtykiet = np.delete(listaTestujaca, len(listaTestujaca[0])-1, axis=1)

#print(sp.distance.euclidean(listaUczaca[0], listaTestujaca[0]))

artificialIntelligence = ClasskNN(3, listaUczaca)
