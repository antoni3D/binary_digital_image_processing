import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu
imageGrace = cv2.imread("obrazy/grace_k.bmp", cv2.IMREAD_GRAYSCALE)
imageCien = cv2.imread("obrazy/cien.bmp", cv2.IMREAD_GRAYSCALE)
imagemb1 = cv2.imread("obrazy/mb1.bmp", cv2.IMREAD_GRAYSCALE)
imagemb2 = cv2.imread("obrazy/mb2.bmp", cv2.IMREAD_GRAYSCALE)
imagebialy = cv2.imread("obrazy/img.png", cv2.IMREAD_GRAYSCALE)


def zadanie1(image):
    # Filtracja medianowa dolnoprzepustowa
    filtered_image = cv2.medianBlur(image, 5)  # Rozmiar okna filtracji: 5x5

    # Wyświetlenie obrazu oryginalnego i przefiltrowanego
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Obraz oryginalny')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Obraz po filtracji medianowej')
    plt.axis('off')

    plt.show()


def zadanie2(image):
    # Filtracja medianowa dolnoprzepustowa
    filtered_image = cv2.medianBlur(image, 5)  # Rozmiar okna filtracji: 5x5

    # Filtracja medianowa górnoprzepustowa
    high_pass_image = image - filtered_image

    # Wyświetlenie obrazu oryginalnego i przefiltrowanego
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Obraz oryginalny')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(high_pass_image, cmap='gray')
    plt.title('Obraz po filtracji medianowej górnoprzepustowej')
    plt.axis('off')
    plt.show()


def zadanie3(image1, image2):
    # Wyznaczenie średniej arytmetycznej obu obrazów
    S = (image1 + image2) / 2.0

    # Odejmowanie obrazu średniego od obrazu A, aby otrzymać szum
    # Dodajemy 0.5, aby uniknąć ujemnych wartości pikseli
    SZUM = image1 - S + 0.5

    # Wyświetlanie obrazów
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Obraz A')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(SZUM, cmap='gray')
    plt.title('SZUM')
    plt.axis('off')

    plt.show()

    # Wyświetlanie powierzchni obrazów
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(0, image1.shape[1])
    Y = np.arange(0, image1.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax1.plot_surface(X, Y, image1, cmap='viridis')
    ax1.set_title('Powierzchnia obrazu A')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, SZUM, cmap='viridis')
    ax2.set_title('Powierzchnia obrazu SZUM')

    plt.show()


def zadanie4(prog, okno):
    # Wczytanie obrazów
    A = cv2.imread("obrazy/ma010.bmp", cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    B = cv2.imread("obrazy/ma011.bmp", cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

    # Obliczenie różnicy i dodanie 0.5
    C = B - A + 0.5

    # Normalizacja
    MinC = np.min(C)
    MaxC = np.max(C)
    C = (C - MinC) / (MaxC - MinC)

    # Binaryzacja progowa
    D = (C > prog).astype(np.uint8)

    # Filtracja medianowa
    D = cv2.medianBlur(D, okno)

    # Wyświetlenie obrazów
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(A, cmap='gray')
    plt.title('A = ' + "pierwszy obraz")

    plt.subplot(2, 2, 2)
    plt.imshow(B, cmap='gray')
    plt.title('B = ' + "drugi obraz")

    plt.subplot(2, 2, 3)
    plt.imshow(C, cmap='gray')
    plt.title('C = B - A + 0.5 (po normalizacji)')

    plt.subplot(2, 2, 4)
    plt.imshow(D, cmap='gray')
    plt.title('Po binaryzacji, PROG = ' + str(prog))

    # Trójwymiarowy obraz powierzchni obrazu C
    plt.figure()
    plt.imshow(C, cmap='gray')
    plt.title('Obraz C')
    plt.colorbar()

    plt.show()


def zadanie8():
    def edge_detection(image):
        # Filtr Prewitta
        prewitt_x = cv2.filter2D(image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        prewitt_combined = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))

        # Filtr Sobela
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

        # Filtr Robertsa
        roberts_x = cv2.filter2D(image, -1, np.array([[1, 0], [0, -1]]))
        roberts_y = cv2.filter2D(image, -1, np.array([[0, 1], [-1, 0]]))
        roberts_combined = np.sqrt(np.square(roberts_x) + np.square(roberts_y))

        return prewitt_combined, sobel_combined, roberts_combined

    # Przykładowe użycie funkcji
    image = cv2.imread("obrazy/img.png", cv2.IMREAD_GRAYSCALE)

    prewitt_edges, sobel_edges, roberts_edges = edge_detection(image)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Prewitt Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(roberts_edges, cmap='gray')
    plt.title('Roberts Edge Detection')
    plt.axis('off')

    plt.show()


zadanie1(imageGrace)
zadanie2(imageCien)
zadanie3(imagemb1,imagemb2)
zadanie4(.75, 3)
zadanie8()
