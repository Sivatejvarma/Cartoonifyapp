import cv2
import easygui
import numpy as np
import tkinter as tk
from tkinter import filedialog, Button, TOP
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class CartoonifyApp:
    def __init__(self):
        self.top = tk.Tk()
        self.top.geometry('800x400')
        self.top.title('Cartoonify Your Image!')
        self.top.configure(background='white')
        self.label = tk.Label(self.top, background='#CDCDCD', font=('calibri', 20, 'bold'))
        self.label.pack(side=tk.LEFT)
        self.upload_button = Button(self.top, text="Cartoonify an Image", command=self.upload, padx=10, pady=5)
        self.upload_button.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
        self.upload_button.pack(side=TOP, pady=50)

    def run(self):
        self.top.mainloop()

    def upload(self):
        ImagePath = easygui.fileopenbox()
        self.cartoonify(ImagePath)

    def save_image(self, image, name, path):
        cv2.imwrite(os.path.join(path, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def cartoonify(self, ImagePath):
        originalImage = cv2.imread(ImagePath)
        originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

        if originalImage is None:
            print("Can not find any image. Choose an appropriate file")
            return

        # Create a directory to save individual images
        save_path = os.path.splitext(ImagePath)[0] + "_cartoon_images"
        os.makedirs(save_path, exist_ok=True)

        # Resize the image for faster processing
        originalImage = cv2.resize(originalImage, (400, 400))

        self.save_image(originalImage, "original_image.jpg", save_path)

        grayScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
        getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 9)

        self.save_image(grayScaleImage, "gray_scale_image.jpg", save_path)
        self.save_image(smoothGrayScale, "smooth_gray_scale_image.jpg", save_path)
        self.save_image(getEdge, "edges_image.jpg", save_path)

        colorImage = cv2.bilateralFilter(originalImage, 9, 300, 300)

        # Additional cartoonify steps from the second program
        line_wdt = 9
        blur_value = 7
        totalColors = 4

        edgeImg = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, line_wdt, blur_value)

        img_quantized = self.color_quantisation(colorImage, totalColors)
        blurred = cv2.bilateralFilter(img_quantized, d=7, sigmaColor=200, sigmaSpace=200)
        cartoonImage = cv2.bitwise_and(blurred, blurred, mask=edgeImg)
        cartoonImage = cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB)

        self.save_image(cartoonImage, "cartoonified_image.jpg", save_path)

        # Display images using plt
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(originalImage)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(grayScaleImage, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(getEdge, cmap='gray')
        plt.title('Edges')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(cartoonImage)
        plt.title('Cartoonified Image')
        plt.axis('off')

        plt.show()

    def color_quantisation(self, img, k):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

if __name__ == "__main__":
    app = CartoonifyApp()
    app.run()
