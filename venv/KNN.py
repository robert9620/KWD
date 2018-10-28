import numpy as np
import scipy.spatial as sp
import operator


class KNN:
    def __init__(self, k, baza_danych):
        self.k = k
        self.baza_danych = baza_danych
        self.baza_danych_bez_etykiet = np.delete(baza_danych, len(baza_danych[0])-1, axis=1)
        self.numer_kolumny_etykiet = len(baza_danych[0])-1

    def predict(self, lista_z_obiektami):
        gotowe_etykiety = []
        for y in range(len(lista_z_obiektami)):

            wszystkie_odleglosci = []
            for x in range(len(self.baza_danych)):
                odleglosc = sp.distance.euclidean(self.baza_danych_bez_etykiet[x], lista_z_obiektami[y])
                wszystkie_odleglosci.append([odleglosc,self.baza_danych[x][self.numer_kolumny_etykiet]])

            najblizsze_etykiety = []
            self.sortuj(wszystkie_odleglosci)
            # print(wszystkie_odleglosci)
            for x in range(self.k):
                najblizsze_etykiety.append(wszystkie_odleglosci[x][1])
            #print(najblizsze_etykiety)

            ktorej_etykiety_najwiecej = {}
            for x in range(self.k):
                if najblizsze_etykiety[x] in ktorej_etykiety_najwiecej:
                    ktorej_etykiety_najwiecej[najblizsze_etykiety[x]] += 1
                else:
                    ktorej_etykiety_najwiecej[najblizsze_etykiety[x]] = 1
            #print(ktorej_etykiety_najwiecej)
            posortowane_etykiety = sorted(ktorej_etykiety_najwiecej, key=ktorej_etykiety_najwiecej.get, reverse=True)
            #print(posortowane_etykiety)

            gotowe_etykiety.append(posortowane_etykiety[0])

        return gotowe_etykiety

    def score(self, lista_z_obiektami, lista_etykiet):
        gotowe_etykiety = self.predict(lista_z_obiektami)
        wynik=0
        for x in range (len(lista_z_obiektami)):
            if gotowe_etykiety[x] == lista_etykiet[x]:
                wynik+=1
        return str(wynik*100/len(lista_z_obiektami))

    def sortuj(self, do_sortowania):
        for i in range(len(do_sortowania) - 1, 0, -1):
            for j in range(i):
                if do_sortowania[j][0] > do_sortowania[j + 1][0]:
                    do_sortowania[j][0], do_sortowania[j + 1][0] = do_sortowania[j + 1][0], do_sortowania[j][0]
                    do_sortowania[j][1], do_sortowania[j + 1][1] = do_sortowania[j + 1][1], do_sortowania[j][1]
        return do_sortowania