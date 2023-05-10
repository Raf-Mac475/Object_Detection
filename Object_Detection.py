import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Object Detection")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

# create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=0)

        #Logo
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Object Detection", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        #Upload Button
        self.upload_picture_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.upload_picture_btn_event)
        self.upload_picture_btn.grid(row=1, column=0, padx=20, pady=10)

        #Take picture Button
        self.take_picture_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.take_picture_btn_event)
        self.take_picture_btn.grid(row=2, column=0, padx=20, pady=10)

        #Camera Button
        self.camera_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.camera_btn_event)
        self.camera_btn.grid(row=3, column=0, padx=20, pady=10)

        #Model Label
        self.model_label = customtkinter.CTkLabel(self.sidebar_frame, text="Choose Model:", anchor="w")
        self.model_label.grid(row=4, column=0, padx=20, pady=(10, 0))

        #Model choosing option menu
        self.model_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Yolov7", "DeepLabV3"],
                                                                       command=self.change_model_event)
        self.model_optionmenu.grid(row=5, column=0, padx=20, pady=(10, 10))

        #Appearance and UI Label
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode and UI Scaling:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(150, 0))

        #Appearance option menu
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        
        #UI Scaling option menu
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))


#Picture  mat creation
        self.picture_mat= customtkinter.CTkLabel(self, text=" ")
        self.picture_mat.grid(row=1, column=1)
# Adding functionality to zoom in and out
        self.picture_mat.bind("<MouseWheel>", self.zoom_image)

        # set default values
        self.upload_picture_btn.configure(text="Upload Picture")
        self.take_picture_btn.configure(text="Take picture")
        self.camera_btn.configure(text="Camera")
        self.appearance_mode_optionemenu.set("System")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_model_event(self, new_model):
        print(new_model)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def upload_picture_btn_event(self):
        print("upload_button click")
        self.upload_picture()

    def take_picture_btn_event(self):
        print("picture_button click")
        self.Camera_photo()

    def camera_btn_event(self):
        print("camera_button click")
        self.Camera_capture()


    def Camera_capture(self):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                #exit()
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Display the resulting frame
                cv2.imshow('frame', gray)
                if cv2.waitKey(1) == ord('q'):
                    break

    def Camera_photo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
        while True:
            # reading the input using the camera
            result, image = cap.read()
            # If image will detected without any error,
            # show result
            if not result:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if result:
                # saving image in local storage
                cv2.imwrite("result.jpg", image)

                # output
                img = Image.open("result.jpg")
                my_image = customtkinter.CTkImage(light_image=img, size=(img.width,img.height))
                self.picture_mat.configure(image=my_image)

    # Zoom In function
    def zoom_image(self, event):
        img = Image.open('result.jpg')
        delta = event.delta
        factor = 1.1 if delta > 0 else 0.9
        current_width = self.picture_mat.winfo_width()
        current_height = self.picture_mat.winfo_height()
        new_width = int(current_width * factor)
        new_height = int(current_height * factor)
        show_image = customtkinter.CTkImage(light_image=img, size=(new_width,new_height))
        self.picture_mat.configure(image=show_image)
        self.picture_mat.focus_set()

    def upload_picture(self):
        try:
            filename = filedialog.askopenfilename(initialdir="/Pictures", title="Select a File", filetypes=(("All Files", "*.*"),
            ("jpeg files", "*.jpeg"), ("png files", "*.png") ))
            img = Image.open(filename)
            img.save('result.jpg', "JPEG")
            show_image = customtkinter.CTkImage(light_image=img, size=(img.width,img.height))
            self.picture_mat.configure(image=show_image)
            self.picture_mat.focus_set()
        except:
            print("No file chosen")


        
if __name__ == "__main__":
    app = App()
    app.mainloop()
