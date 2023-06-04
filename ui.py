import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import model_inference as mi
import cv2

class App:
    # initialize application
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Detector")
        self.root.geometry("400x300")

        self.model = mi.load_ml_model()
        self.canvas = None
        # add label and button
        load_label = tk.Label(self.root, text="Load Ship Image:")
        load_label.pack()

        load_button = tk.Button(self.root, text="Load Image", command=self.load_file)
        load_button.pack()

    def load_file(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        # load file
        file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
        # refer to inference
        image, resized_image = mi.preprocessing(file_path)
        predicted_masks = mi.find_segment(self.model, [image])
        ship_detected_result = mi.detect_ship(predicted_masks.squeeze())
        # display graph
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_masks.squeeze(), cmap='gray')
        plt.title(f'Predicted Mask\n{ship_detected_result}')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

# start an app
root = tk.Tk()
app = App(root)
root.mainloop()