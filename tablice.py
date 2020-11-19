
import cv2 as cv
import pytesseract
import glob
import os

def findindexes(lista, limit):
    new_lista_indexes = []
    for i in range(len(lista)):
        if lista[i] > limit:
            new_lista_indexes.append(i)
    return new_lista_indexes
    
def histo_cut (zdj):
    #print(zdj.shape)
    x = zdj.shape[0]
    y = zdj.shape[1]
    """print('wiersze (x): ', x)
    print('kolumny (y): ', y)"""
    # kolumny
    kolumny = []
    for b in range(y):
        j = sum(zdj[:, b])
        kolumny.append(j)

    # wiersze
    wiersze = []
    for y in range(x):
        c = sum(zdj[y, :])
        wiersze.append(c)

    """ print('sumy z kolumn: ', kolumny)
    print('sumy z wierszy: ', wiersze)

    print('srednia kolumny: ', sum(kolumny) / float(len(kolumny)))
    print('srednia wiersze: ', sum(wiersze) / float(len(wiersze)))"""

    factor = 1.1
    cut_kolumny = findindexes(kolumny, factor * (sum(kolumny) / float(len(kolumny))))
    cut_wiersze = findindexes(wiersze, factor * (sum(wiersze) / float(len(wiersze))))
    """print(cut_kolumny)
    print(cut_wiersze)"""


    
    # roi_cut = img[0:46,32:120]
    cut = zdj[cut_wiersze[0]:cut_wiersze[-1], cut_kolumny[0]:cut_kolumny[-1]]
    """cv.imshow('cut', cut)
    cv.waitKey(0)

    plt.show()"""
    return cut_wiersze[0],cut_wiersze[-1], cut_kolumny[0],cut_kolumny[-1]


for file in glob.glob("*.jpg"):

    
    tablica = cv.CascadeClassifier('haarcascade_russian_plate_number.xml')
    img = cv.imread(file)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('odcienie szarosci', gray)
    #cv.waitKey(0)
    samochod = tablica.detectMultiScale(gray,1.1,20) #szukanie tablicy. Jeśli znajdzie to zwraca (x,y,w,h)


    #print ('znalezione przez detectMultiScale',samochod)
   
    for (x, y, w, h) in samochod:  # przeprowadzamy iteracje po znalezionym wzorcu
        print('nazwa aktualnego pliku to: ',file)
        roi_gray = gray[y:y + h+20, x:x + w]
        # binetalizacja obrazu oraz przekazanie do tworzenia histogramu brzegowego
        cv.imshow('znaleziona tablica',roi_gray)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        ret, thresh = cv.threshold(roi_gray, 120, 255, cv.THRESH_BINARY)
        
        x1,x2,y1,y2 = histo_cut(thresh)
        #print(x1,x2,y1,y2)
        histo =roi_gray[x1:x2,y1:y2]
        #cv.imshow('histogram',histo)
        #cv.imshow('roi_gray', roi_gray)
        text = pytesseract.image_to_string(histo, config='--psm 6')
        print('tekst na tablicy to: ',text)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
       
