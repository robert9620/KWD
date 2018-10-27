import numpy as np
import pandas as pd

class kNN:
    def __init__(self, k, listaUczaca):
        self.k = k

    def predict(self, listaZObiektami):
        return ""

    def score(self, listaZObiektami, listaEtykiet):
        return ""


listaUczaca = pd.read_csv("data-learning.csv")
listaTestujaca = pd.read_csv("data-test.csv")
print(listaTestujaca)
