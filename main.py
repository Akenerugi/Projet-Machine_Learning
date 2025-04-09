import numpy as np
import joblib
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw

class DrawingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Test Dessin")
        self.geometry("800x1000")


        self.lettre_binaires = {
            (0, 0, 0): "B",
            (0, 0, 1): "F",
            (0, 1, 0): "G",
            (0, 1, 1): "H",
            (1, 0, 0): "J",
            (1, 0, 1): "K"
        }


        self.frame = ctk.CTkFrame(self, width=780, height=60, corner_radius=10)
        self.frame.pack(padx=5, pady=35)
        self.label1 = ctk.CTkLabel(self.frame, text="PROJET ML", font=("ITC Avant Garde Gothic LT Bold", 28))
        self.label1.pack(padx=5, pady=10)

        # Frame pour le canvas
        self.frame3 = ctk.CTkFrame(self, width=780, height=150, corner_radius=10)
        self.frame3.pack(padx=15, pady=10)

        self.label3 = ctk.CTkLabel(self.frame3, text="Dessinez ici :", font=("ITC Avant Garde Gothic LT Bold", 28))
        self.label3.pack(padx=5, pady=10)

        # Canvas pour dessiner
        self.canvas = tk.Canvas(self.frame3, bg="white", width=512, height=512)
        self.canvas.pack(padx=15, pady=10)

        # Frame pour les boutons
        self.frame_buttons = ctk.CTkFrame(self, width=780, height=60, corner_radius=10)
        self.frame_buttons.pack(padx=10, pady=10)

        self.bouttondelete = ctk.CTkButton(
            self.frame_buttons, text="Effacer", font=("ITC Avant Garde Gothic LT Bold", 28), command=self.delete_canvas
        )
        self.bouttondelete.grid(row=0, column=0, padx=5, pady=10)

        self.bouttonsave = ctk.CTkButton(
            self.frame_buttons, text="Enregistrer", font=("ITC Avant Garde Gothic LT Bold", 28), command=self.save_image
        )
        self.bouttonsave.grid(row=0, column=1, padx=5, pady=10)

        self.bouttonscan = ctk.CTkButton(
            self.frame_buttons, text="Scan", font=("ITC Avant Garde Gothic LT Bold", 28), command=self.scan_image
        )
        self.bouttonscan.grid(row=0, column=2, padx=5, pady=10)


        self.prediction_label = ctk.CTkLabel(self, text="Lettre prédite : ", font=("ITC Avant Garde Gothic LT Bold", 28))
        self.prediction_label.pack(padx=5, pady=10)


        self.init_image()


        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

    def init_image(self):

        self.image = Image.new("RGB", (512, 512), "white")
        self.draw = ImageDraw.Draw(self.image)

    def draw_on_canvas(self, event):

        if hasattr(self, 'old_x') and hasattr(self, 'old_y') and self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=2, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill="black", width=2)
        
        self.old_x, self.old_y = event.x, event.y

    def reset_position(self, event):

        self.old_x, self.old_y = None, None

    def delete_canvas(self):

        self.canvas.delete("all")
        self.init_image()
        print("Canvas réinitialisé.")

    def save_image(self):

        file_path = "dessin_canvas.jpeg"
        self.image.save(file_path, "JPEG")
        print(f"Image sauvegardée sous le nom {file_path}")

    def scan_image(self):

        coefficients = self.process_image_grid(self.image)
        predicted_label = self.predict_new_image(coefficients)

        self.prediction_label.configure(text=f"Lettre prédite : {predicted_label}")

    def process_image_grid(self, image):

        grid_size = 8
        cell_width = image.width // grid_size
        cell_height = image.height // grid_size
        black_pixel_coordinates = []

        for row in range(grid_size):
            for col in range(grid_size):
                x_start, y_start = col * cell_width, row * cell_height
                x_end, y_end = x_start + cell_width, y_start + cell_height
                cell_coords = []

                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if image.getpixel((x, y))[0] < 128:
                            relative_x = x - x_start
                            relative_y = y - y_start
                            cell_coords.append((relative_x, relative_y))

                if len(cell_coords) > 0:
                    X_train = np.array([x[0] for x in cell_coords]).reshape(-1, 1)
                    y_train = np.array([x[1] for x in cell_coords]).reshape(-1, 1)

                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression().fit(X_train, y_train)
                    coefficient = reg.coef_[0][0]
                    black_pixel_coordinates += [(2 * coefficient) / (1 + coefficient**2), (1 - coefficient**2) / (1 + coefficient**2)]
                else:
                    black_pixel_coordinates += [0, 0]

        return black_pixel_coordinates

    def predict_new_image(self, coefficients):

        try:
            mlp = joblib.load("mlp_model_3couches.pkl")
            coefficients = np.array(coefficients).reshape(1, -1)
            prediction = tuple(mlp.predict(coefficients)[0])

            return self.lettre_binaires.get(prediction, "Inconnu")
        except Exception as e:
            return f"Erreur : {str(e)}"

if __name__ == "__main__":
    app = DrawingApp()
    app.mainloop()
