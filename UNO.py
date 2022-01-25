import os
import cv2
import numpy as np
import glob

from numpy import copysign, log10


def przycinanie(img):
    wpolrzedne = []
    contorus, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contorus:
        area = cv2.contourArea(cnt)
        if area > 30000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            a = int(x + w / 2)
            b = int(y + h / 2)

            c = int(a - 90)
            d = int(b - 90)

            wpolrzedne.append(c)
            wpolrzedne.append(d)
    return wpolrzedne


def srodek(img):
    wpolrzedne = []
    contorus, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contorus:
        area = cv2.contourArea(cnt)
        if area > 30000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            a = int(x + w / 2)
            b = int(y + h / 2)

            wpolrzedne.append(a)
            wpolrzedne.append(b)
    return wpolrzedne


def kontur(img, imgKontur):
    contorus, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pola = []
    b = 0
    for cnt in contorus:
        area = cv2.contourArea(cnt)
        pola.append(area)

    for p in range(len(pola)):
        if pola[p] > 1000:
            cv2.drawContours(imgKontur, contorus, p, (255, 0, 255), 7)
            a = cv2.moments(contorus[p])
            b = cv2.HuMoments(a)
    return b


def wartosci(a):
    wartosc = 0
    if 0.721 < a[0][0] < 0.736:
        wartosc = 4
    if 0.629 < a[0][0] < 0.668:
        wartosc = 2
    if 0.768 < a[0][0] < 0.785:
        wartosc = 0
    if 0.79 < a[0][0] < 0.793:
        wartosc = 11
    if 0.593 < a[0][0] < 0.618:
        wartosc = 12

    return wartosc


def maski(rodzaj_maski, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 150, 90])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([15, 150, 80])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([35, 120, 90])
    upper_green = np.array([65, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_red = np.array([0, 100, 120])
    upper_red = np.array([25, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    suma = 0
    wspolrzedne = []

    if rodzaj_maski == 'niebieski':
        mask = blue_mask
        mask = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 13)
        canny = cv2.Canny(blur, 100, 150)
        lista = przycinanie(canny)
        lista_max = lista.copy()
        zs = srodek(canny)
        znak_specjalny = [zs[0], zs[1]]

        for k in range(0, len(lista_max)):
            lista_max[k] = int(lista_max[k] + 170)

        if len(lista) == 4:
            roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            roi_2 = canny[lista[3]:lista_max[3], lista[2]:lista_max[2]]
            roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            roi_2_2 = img[lista[3]:lista_max[3], lista[2]:lista_max[2]]
            a = kontur(roi_1, roi_1_1)
            b = kontur(roi_2, roi_2_2)
            for element in range(0, 7):
                a[element] = -1 * copysign(1.0, a[element]) * log10(abs(a[element]))
                b[element] = -1 * copysign(1.0, b[element]) * log10(abs(b[element]))
            wartosc_a = wartosci(a)
            if wartosc_a > 10:
                wspolrzedne += znak_specjalny
            else:
                suma += wartosc_a
            wartosc_b = wartosci(b)
            if wartosc_b > 10:
                wspolrzedne += znak_specjalny
            else:
                suma += wartosc_b
        else:
            roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            a = kontur(roi_1, roi_1_1)
            for element in range(0, 7):
                a[element] = -1 * copysign(1.0, a[element]) * log10(abs(a[element]))
            wartosc_a = wartosci(a)
            if wartosc_a > 10:
                wspolrzedne = znak_specjalny
            else:
                suma += wartosc_a

    if rodzaj_maski == 'zolty':
        mask = yellow_mask
        mask = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 13)
        canny = cv2.Canny(blur, 100, 150)
        lista = przycinanie(canny)
        lista_max = lista.copy()
        znak_specjalny = srodek(canny)

        for k in range(0, len(lista_max)):
            lista_max[k] = int(lista_max[k] + 170)

        roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
        roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
        a = kontur(roi_1, roi_1_1)
        for element in range(0, 7):
            a[element] = -1 * copysign(1.0, a[element]) * log10(abs(a[element]))
        wartosc_a = wartosci(a)
        if wartosc_a > 10:
            wspolrzedne = znak_specjalny
        else:
            suma += wartosc_a

    if rodzaj_maski == 'zielony':
        mask = green_mask
        mask = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 13)
        canny = cv2.Canny(blur, 100, 150)
        lista = przycinanie(canny)
        lista_max = lista.copy()
        znak_specjalny = srodek(canny)

        for k in range(0, len(lista_max)):
            lista_max[k] = int(lista_max[k] + 170)

        roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
        roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
        a = kontur(roi_1, roi_1_1)
        for element in range(0, 7):
            a[element] = -1 * copysign(1.0, a[element]) * log10(abs(a[element]))
        wartosc_a = wartosci(a)
        if wartosc_a > 10:
            wspolrzedne = znak_specjalny
        else:
            suma += wartosc_a

    if rodzaj_maski == 'czerwony':
        mask = red_mask
        mask = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 13)
        canny = cv2.Canny(blur, 100, 150)
        lista = przycinanie(canny)
        lista_max = lista.copy()
        znak_specjalny = srodek(canny)

        for k in range(0, len(lista_max)):
            lista_max[k] = int(lista_max[k] + 170)

        if len(lista) != 0:
            roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
            a = kontur(roi_1, roi_1_1)
            for element in range(0, 7):
                a[element] = -1 * copysign(1.0, a[element]) * log10(abs(a[element]))
            wartosc_a = wartosci(a)
            if wartosc_a > 10:
                wspolrzedne = znak_specjalny
            else:
                suma += wartosc_a

    return suma, wspolrzedne


def znaki_specjalne(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 150, 90])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([15, 150, 80])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([35, 120, 90])
    upper_green = np.array([65, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_red = np.array([0, 100, 120])
    upper_red = np.array([25, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    mask = blue_mask + yellow_mask + green_mask + red_mask

    mask = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 13)
    canny = cv2.Canny(blur, 100, 150)
    lista = przycinanie(canny)
    lista_max = lista.copy()
    s = srodek(canny)
    w = []

    for k in range(0, len(lista_max)):
        lista_max[k] = int(lista_max[k] + 170)

    roi_1 = canny[lista[1]:lista_max[1], lista[0]:lista_max[0]]
    roi_2 = canny[lista[3]:lista_max[3], lista[2]:lista_max[2]]
    roi_3 = canny[lista[5]:lista_max[5], lista[4]:lista_max[4]]
    roi_4 = canny[lista[7]:lista_max[7], lista[6]:lista_max[6]]

    roi_1_1 = img[lista[1]:lista_max[1], lista[0]:lista_max[0]]
    roi_2_2 = img[lista[3]:lista_max[3], lista[2]:lista_max[2]]
    roi_3_3 = img[lista[5]:lista_max[5], lista[4]:lista_max[4]]
    roi_4_4 = img[lista[7]:lista_max[7], lista[6]:lista_max[6]]

    a = kontur(roi_1, roi_1_1)
    b = kontur(roi_2, roi_2_2)
    c = kontur(roi_3, roi_3_3)
    d = kontur(roi_4, roi_4_4)

    for i in range(0, 7):
        a[i] = -1 * copysign(1.0, a[i]) * log10(abs(a[i]))
        b[i] = -1 * copysign(1.0, b[i]) * log10(abs(b[i]))
        c[i] = -1 * copysign(1.0, c[i]) * log10(abs(c[i]))
        d[i] = -1 * copysign(1.0, d[i]) * log10(abs(d[i]))

    wartosc_a = wartosci(a)
    if wartosc_a > 10:
        w.append([s[0], s[1]])

    wartosc_b = wartosci(b)
    if wartosc_b > 10:
        w.append([s[2], s[3]])

    wartosc_c = wartosci(c)
    if wartosc_c > 10:
        w.append([s[4], s[5]])

    wartosc_d = wartosci(d)
    if wartosc_d > 10:
        w.append([s[6], s[7]])
    return w


file = 'karty\*.png'
images = glob.glob(file)

suma_kart = 0
suma_zielony = 0
suma_czerwony = 0
suma_niebieski = 0
suma_zolty = 0

print('Podaj kolor/kolory kart')
print('Dostepne kolory: czerwony, zolty, zielony, niebieski, wszystkie')
x = input()

for i in range(0, len(images)):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img1 = img.copy()
    print(f'Zdjecie {i+1}')
    zolty = maski('zolty', img)
    suma_zolty = zolty[0]

    niebieski = maski('niebieski', img)
    suma_niebieski = niebieski[0]

    czerwony = maski('czerwony', img)
    suma_czerwony = czerwony[0]

    zielony = maski('zielony', img)
    suma_zielony = zielony[0]

    if x == 'zielony':
        print(f'Suma zielonych kart: {suma_zielony}')

    elif x == 'czerwony':
        print(f'Suma czerownych kart: {suma_czerwony}')

    elif x == 'zolty':
        print(f'Suma zoltych kart: {suma_zolty}')

    elif x == 'niebieski':
        print(f'Suma niebieskich kart: {suma_niebieski}')

    elif x == 'wszystkie':
        suma_kart = suma_niebieski + suma_zolty + suma_zielony + suma_czerwony
        print(f'Suma wszystkich kart: {suma_kart}')
    print(f'Wspolrzedne znaku spechalnego: {znaki_specjalne(img1)}')
