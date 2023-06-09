import customtkinter
import cv2
from PIL import Image
from threading import Thread
from tkinter import filedialog
from datetime import datetime

from generic_segmenter import *


customtkinter.set_appearance_mode('System')  # Modes: 'System' (standard), 'Dark', 'Light'
customtkinter.set_default_color_theme('blue')  # Themes: 'blue' (standard), 'green', 'dark-blue'


class SampleIterator():
    def __init__(self):
        self.iter = 0
    
    def next(self) -> int:
        self.iter += 1
        return self.iter - 1
    
    def restart(self):
        self.iter = 0


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.iterator = SampleIterator()

        # configure window
        self.title('Object Detection')
        self.geometry(f'{1100}x{580}')

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky='nsew')
        self.sidebar_frame.grid_rowconfigure(4, weight=0)

        #Logo
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text='Object Detection', font=customtkinter.CTkFont(size=20, weight='bold'))
        self.logo_label.grid(row=self.iterator.next(), column=0, padx=20, pady=(20, 10))

        #Upload Button
        self.upload_picture_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.upload_picture_btn_event)
        self.upload_picture_btn.grid(row=self.iterator.next(), column=0, padx=20, pady=10)

        #Take picture Button
        self.take_picture_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.take_picture_btn_event)
        self.take_picture_btn.grid(row=self.iterator.next(), column=0, padx=20, pady=10)

        #Camera Button
        self.camera_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.camera_btn_event)
        self.camera_btn.grid(row=self.iterator.next(), column=0, padx=20, pady=10)

        #Model Label
        self.model_label = customtkinter.CTkLabel(self.sidebar_frame, text='Choose Model:', anchor='w')
        self.model_label.grid(row=self.iterator.next(), column=0, padx=20, pady=(10, 0))

        #Model choosing option menu
        self.model_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=['DUMMY_SEGMENTER','DeepLabV3', 'YoloV7','droniada'],
                                                                       command=self.change_model_event)
        self.model_optionmenu.grid(row=self.iterator.next(), column=0, padx=20, pady=(10, 10))
        print(self.model_optionmenu.get())
        self.segmenter = get_segmenter(self.model_optionmenu.get(), './ml_models/DeepLabV3.pt')

        #Segment Button
        self.segment_btn = customtkinter.CTkButton(self.sidebar_frame, command=self.segment_magic)
        self.segment_btn.grid(row=self.iterator.next(), column=0, padx=20, pady=10)

        #Appearance and UI Label
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text='Appearance Mode and UI Scaling:', anchor='w')
        self.appearance_mode_label.grid(row=self.iterator.next(), column=0, padx=20, pady=(100, 0))

        #Appearance option menu
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=['Light', 'Dark', 'System'],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=self.iterator.next(), column=0, padx=20, pady=(10, 10))
        
        #UI Scaling option menu
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=['80%', '90%', '100%', '110%', '120%'],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=self.iterator.next(), column=0, padx=20, pady=(10, 20))


        # Picture  mat creation
        self.picture_mat= customtkinter.CTkLabel(self, text=' ')
        self.picture_mat.grid(row=1, column=1)
        # Adding functionality to zoom in and out
        self.picture_mat.bind('<MouseWheel>', self.zoom_image)

        # set default values
        self.upload_picture_btn.configure(text="Upload Picture")
        self.take_picture_btn.configure(text='Take picture')
        self.camera_btn.configure(text="Camera")
        self.appearance_mode_optionemenu.set("System")
        self.scaling_optionemenu.set("100%")
        self.segment_btn.configure(text='Segment', state='disabled')


    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_model_event(self, new_model):
        print(new_model)
        if new_model == 'DUMMY_SEGMENTER':
            self.segmenter = get_segmenter(new_model, './DUMMY_SEGMENTER.pt')
        elif new_model == 'DeepLabV3':
            self.segmenter = get_segmenter(new_model, './ml_models/DeepLabV3.pt')
        elif new_model == 'YoloV7':
            self.segmenter = get_segmenter(new_model, './ml_models/YoloV7.pt')
        elif new_model == 'droniada':
            self.segmenter = get_segmenter(new_model, './ml_models/droniada_rurociag.pt')
        else:
            raise RuntimeError('How the quack did you do that???')

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace('%', '')) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def upload_picture_btn_event(self):
        print("upload_button click")
        self.upload_picture_from_disk()

    def take_picture_btn_event(self):
        print("picture_button click")
        self.Camera_photo()


    def camera_btn_event(self):
        print('camera_button click')
        self.Camera_capture()
    
    # TODO(jakubg): remember to block all functionalities that shouldn't be changed during segmentation.
    def block_segment_btn(self):
        self.segment_btn.configure(state="disabled")

    def unlock_segment_btn(self):
        self.segment_btn.configure(state="normal")

    def do_segmentation(self):
        segmented_image= self.segmenter.segment(self.my_image._light_image)
        self.my_image = customtkinter.CTkImage(light_image=segmented_image,
                                               size=(self.img.width, self.img.height))
        self.picture_mat.configure(image=self.my_image)

        self.unlock_segment_btn()
        self.save_result(segmented_image)
        print('Ended')

    def save_result(self, image):
        now = datetime.now()  # current date and time
        date_time = str(now.strftime("Result_%m%d%Y%H%M%S"))
        image.save('Results/{0}.png'.format(date_time), "PNG")

    def segment_magic(self):
        print('segment_magic click')
        if self.segment_btn._state != 'disabled':
            self.block_segment_btn()
            print('Started')
            thread = Thread(target=self.do_segmentation)
            thread.start()

        else:
            print('Button is disabled activate it adding image :/ - or program it to do more!')

    def Camera_capture(self):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print('Cannot open camera')
                #exit()
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print('Can\'t receive frame (stream end?). Exiting ...')
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
            print('Cannot open camera')
            return
            
        # reading the input using the camera
        result, image = cap.read()

        if not result:
            print('Can\'t receive frame (stream end?). Exiting ...')
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.upload_picture(img)

    # Zoom In function
    def zoom_image(self, event):
        # img = Image.open('result.jpg')
        delta = event.delta
        factor_dif = 0.05
        self.factor += self.factor * factor_dif if delta > 0 else  self.factor * -factor_dif
        new_width = int(self.img.width * self.factor)
        new_height = int(self.img.height * self.factor)
        show_image = customtkinter.CTkImage(light_image=self.my_image._light_image, size=(new_width, new_height))
        self.picture_mat.configure(image=show_image)
        self.picture_mat.focus_set()

    def upload_picture_from_disk(self):
        try:
            filename = filedialog.askopenfilename(initialdir='/Pictures', title='Select a File', filetypes=(('All Files', '*.*'),
            ('jpeg files', '*.jpeg'), ('png files', '*.png') ))
            img = Image.open(filename)
            self.upload_picture(img)

        except Exception as e:
            print(f'Error: {e}')

    def upload_picture(self, image):
        self.factor = 1.0
        try:
            self.img = image
            self.my_image = customtkinter.CTkImage(light_image=self.img, size=(self.img.width, self.img.height))
            self.picture_mat.configure(image=self.my_image)
            self.picture_mat.focus_set()
            self.segment_btn.configure(state='normal')

        except Exception as e:
            print(f'Error: {e}')

        
if __name__ == '__main__':
    app = App()
    app.mainloop()
