import os
import cv2
import psutil
import threading
import numpy as np
import customtkinter
from PIL import Image
from tkinter import filedialog, messagebox

customtkinter.set_default_color_theme("dark-blue")

class YOLODetector:
    """
    YOLODetector class for performing object detection using YOLOv3.
    """

    def __init__(self):
        """
        Initialize the YOLO Face Detection App.
        """
        self.weights_path = 'yolo/yolov3.weights'
        self.cfg_path = 'yolo/yolov3.cfg'
        self.names_path = 'yolo/coco.names'

        self.net = None
        self.layer_names = None
        self.output_layers = None
        self.classes = None

        self.initiate_yolo()

    def initiate_yolo(self):
        """
        Load YOLO model if all required files are provided.

        This method loads the YOLO model configuration and weights,
        and reads the class names from the provided files.
        """
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
            except Exception as e:
                raise Exception(f"Failed to Initiate YOLO, Make sure the weights, cfg and names are in sub-directory 'yolo' in the script directory: {e}")

    def run(self, image):
        """
        Apply YOLO object detection to the provided image.

        Args:
            image (numpy.ndarray): The image on which object detection is to be applied.

        Returns:
            numpy.ndarray: The image with detected objects highlighted.
        """
        if not self.net:
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")
            return image

        height, width, channels = image.shape

        # Prepare the image for YOLO model
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Process YOLO detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if (x, y, w, h) and all(isinstance(coord, int) for coord in [x, y, w, h]):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the image
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                if all(isinstance(coord, int) for coord in [x, y, w, h]):
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image


class FusionFrame(customtkinter.CTk):
    """
    FusionFrame class for managing the main application window and interactions.
    """
    
    def __init__(self):
        """
        Initialize the FusionFrame application.
        """
        super().__init__()

        self.thread = None
        self.running = False
        self.image_path = None
        self.video_path = None
        self.video_capture = None
        self.loaded_video = None
        self.current_filter = None
        self.detect_objects_flag = False
        
        try:
            self.detector = YOLODetector()
        except Exception as e:
            messagebox.showerror("Failed To Load YOLO", e)
            pid = os.getpid()
            system = psutil.Process(pid)
            system.terminate()

        self.title("FusionFrame - Aamir")
        self.geometry("900x500")
        self.resizable(False, False)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "main_logo.png")), size=(26, 26))
        self.main_cover_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "main_cover_image.png")), size=(500, 150))

        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text=" FusionFrame", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_frame_btn = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Source Selection",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.home_frame_btn_event)
        self.home_frame_btn.grid(row=1, column=0, sticky="ew")

        self.object_detection_frame_btn = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Object Detection",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.object_detection_frame_btn_event)
        self.object_detection_frame_btn.grid(row=2, column=0, sticky="ew")

        self.filters_frame_btn = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Filters List",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.filters_frame_btn_event)
        self.filters_frame_btn.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Dark Theme", "Light Theme", "System Theme"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_cover = customtkinter.CTkLabel(self.home_frame, text="", image=self.main_cover_image)
        self.home_frame_cover.grid(row=0, column=0, pady=50)

        self.h_frame_pic_btn = customtkinter.CTkButton(self.home_frame, text="Load Saved Picture", command=self.h_frame_pic_btn_handler)
        self.h_frame_pic_btn.grid(row=1, column=0, padx=60, pady=15)
        self.h_frame_vid_btn = customtkinter.CTkButton(self.home_frame, text="Load Saved Video", command=self.h_frame_vid_btn_handler)
        self.h_frame_vid_btn.grid(row=2, column=0, padx=20, pady=15)
        self.h_frame_cam_btn = customtkinter.CTkButton(self.home_frame, text="Open Live Camera", command=self.h_frame_cam_btn_handler)
        self.h_frame_cam_btn.grid(row=3, column=0, padx=20, pady=15)

        self.object_detection_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.object_detection_frame.grid_rowconfigure(2, weight=1)
        self.object_detection_frame.grid_columnconfigure(0, weight=1)
        
        self.object_detection_btn = customtkinter.CTkButton(self.object_detection_frame, text="Run YOLO", command=self.object_detection_btn_event)
        self.object_detection_btn.grid(row=0, column=0, pady=(10, 0))

        self.object_detection_panel = customtkinter.CTkLabel(self.object_detection_frame, text="")
        self.object_detection_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))

        self.filters_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.filters_frame.grid_rowconfigure(2, weight=1)
        self.filters_frame.grid_columnconfigure(0, weight=1)
        self.filters_menu = customtkinter.CTkOptionMenu(self.filters_frame, values=["Select Filter", "Gaussian Blur", "Median Blur", "Bilateral Filter", "Canny Edge Detection", "Sobel", "Laplacian", "Adaptive Threshold", "Histogram Equalization", "Sharpen", "Emboss", "Sepia"], command=self.toggle_filter_handler)
        self.filters_menu.grid(row=0, column=0, pady=(10, 0))
        
        self.filters_panel = customtkinter.CTkLabel(self.filters_frame, text="")
        self.filters_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))

        self.select_frame_by_name("home")

    def toggle_filter_handler(self, filter_name): 
        """
        Handle filter selection from the filters menu.

        Args:
            filter_name (str): The name of the selected filter.
        """
        self.current_filter = filter_name
        if self.image_path:  # If an image is loaded, apply the filter
            self.load_image()

    def apply_selected_filter(self, frame, filter_name):
        """
        Apply the selected filter to the given frame.

        Args:
            frame (numpy.ndarray): The frame to which the filter is to be applied.
            filter_name (str): The name of the filter to apply.

        Returns:
            numpy.ndarray: The frame with the applied filter.
        """
        # Blur Filters
        if filter_name == "Gaussian Blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_name == "Median Blur":
            return cv2.medianBlur(frame, 5)
        elif filter_name == "Bilateral Filter":
            return cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Edge Detection Filters
        elif filter_name == "Canny Edge Detection":
            return cv2.Canny(frame, 100, 200)
        elif filter_name == "Sobel":
            sobel = cv2.Sobel(frame, cv2.CV_64F, 1, 1, ksize=5)
            return cv2.convertScaleAbs(sobel)
        elif filter_name == "Laplacian":
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            return cv2.convertScaleAbs(laplacian)
        
        # Thresholding Filters
        elif filter_name == "Adaptive Threshold":
            return cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Histogram Equalization
        elif filter_name == "Histogram Equalization":
            if len(frame.shape) == 2:
                return cv2.equalizeHist(frame)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            return frame
        
        # Sharpen Filter
        elif filter_name == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(frame, -1, kernel)
        
        # Other Filters
        elif filter_name == "Emboss":
            kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            return cv2.filter2D(frame, -1, kernel)
        elif filter_name == "Sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])
            sepia = cv2.transform(frame, sepia_filter)
            sepia = np.clip(sepia, 0, 255)
            return sepia.astype(np.uint8)
        
        return frame

    def stop_all_threads(self):
        """
        Stop all running threads and reset parameters.
        """
        self.running = False
        self.image_path = None
        self.video_path = None
        self.detect_objects_flag = False

        self.filters_panel.configure(image='')
        self.object_detection_panel.configure(image='')
        self.object_detection_btn.configure(text="Run YOLO")

        if self.thread and self.thread.is_alive():
            self.thread = None
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if self.loaded_video is not None: 
            self.loaded_video.release()
            self.loaded_video = None

    def load_image(self):
        """
        Load the selected image and display it, maintaining the aspect ratio.
        """
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect objects in the image
        if self.detect_objects_flag: 
            image = self.detector.run(image)

        # Apply the selected filter
        if self.current_filter:
            image = self.apply_selected_filter(image, self.current_filter)

        original_width, original_height = image.shape[1], image.shape[0]
        max_width, max_height = 720, 420
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = min(max_width, original_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, original_height)
            new_width = int(new_height * aspect_ratio)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        ctk_image = customtkinter.CTkImage(image, size=(new_width, new_height))

        self.object_detection_panel.configure(image=ctk_image)
        self.object_detection_panel.image = ctk_image
        
        self.filters_panel.configure(image=ctk_image)
        self.filters_panel.image = ctk_image

    def h_frame_pic_btn_handler(self):
        """
        Select an image file.
        """
        self.stop_all_threads()
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif")])
        if self.image_path:
            self.load_image()

    def is_source_selected(self):
        """
        Check if a source (image, video, or live) has been selected.

        Returns:
            bool: True if a source is selected, False otherwise.
        """
        return self.image_path or self.video_path or self.running

    def detect_objects_video(self):
        """
        Detect objects in the selected video file and display it in the object detection and filters panels.
        """
        self.loaded_video = cv2.VideoCapture(self.video_path)
        max_width, max_height = 720, 420

        while self.running:
            ret, frame = self.loaded_video.read()
            if not ret:
                break

            if self.detect_objects_flag:
                frame = self.detector.run(frame)

            # Apply the selected filter
            if self.current_filter:
                frame = self.apply_selected_filter(frame, self.current_filter)
            
            original_width, original_height = frame.shape[1], frame.shape[0]
            aspect_ratio = original_width / original_height

            if original_width > original_height:
                new_width = min(max_width, original_width)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(max_height, original_height)
                new_width = int(new_height * aspect_ratio)

            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            ctk_image = customtkinter.CTkImage(img, size=(new_width, new_height))

            self.object_detection_panel.configure(image=ctk_image)
            self.object_detection_panel.image = ctk_image
            self.object_detection_panel.grid(row=1, column=0, padx=10, pady=(10, 20))

            self.filters_panel.configure(image=ctk_image)
            self.filters_panel.image = ctk_image
            self.filters_panel.grid(row=1, column=0, padx=10, pady=(10, 20))

    def h_frame_vid_btn_handler(self):
        """
        Select a video file.
        """
        self.stop_all_threads()
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.flv;*.wmv")])
        if self.video_path:
            self.running = True
            self.thread = threading.Thread(target=self.detect_objects_video)
            self.thread.start()

    def show_live_video(self):
        """
        Display live video feed from the webcam.
        """
        self.video_capture = cv2.VideoCapture(0)
        max_width, max_height = 720, 420

        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # Detect objects
            if self.detect_objects_flag:
                frame = self.detector.run(frame)

            # Apply the selected filter
            if self.current_filter:
                frame = self.apply_selected_filter(frame, self.current_filter)
            
            original_width, original_height = frame.shape[1], frame.shape[0]
            aspect_ratio = original_width / original_height

            if original_width > original_height:
                new_width = min(max_width, original_width)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(max_height, original_height)
                new_width = int(new_height * aspect_ratio)

            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            ctk_image = customtkinter.CTkImage(img, size=(new_width, new_height))

            self.object_detection_panel.configure(image=ctk_image)
            self.object_detection_panel.image = ctk_image
            self.object_detection_panel.grid(row=1, column=0, padx=10, pady=(10, 20))

            self.filters_panel.configure(image=ctk_image)
            self.filters_panel.image = ctk_image
            self.filters_panel.grid(row=1, column=0, padx=10, pady=(10, 20))

    def h_frame_cam_btn_handler(self):
        """
        Start or stop the live video feed.
        """
        self.stop_all_threads()
        self.running = True
        self.thread = threading.Thread(target=self.show_live_video)
        self.thread.start()

    def object_detection_btn_event(self):
        """
        Toggle object detection on/off.
        """
        self.detect_objects_flag = not self.detect_objects_flag
        if self.detect_objects_flag:
            self.object_detection_btn.configure(text="Quit YOLO")
            if self.image_path: 
                self.load_image()
        else:
            self.object_detection_btn.configure(text="Run YOLO")
            if self.image_path: 
                self.load_image()

    def select_frame_by_name(self, name):
        """
        Select the frame by name and handle source selection warning.

        Args:
            name (str): The name of the frame to select.
        """
        if name in ["object_detection", "filters_frame"] and not self.is_source_selected():
            messagebox.showwarning("Source Not Selected", "Please select a source (image, video, or live feed) before switching to this frame.")
            return
        self.home_frame_btn.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.object_detection_frame_btn.configure(fg_color=("gray75", "gray25") if name == "object_detection" else "transparent")
        self.filters_frame_btn.configure(fg_color=("gray75", "gray25") if name == "filters_frame" else "transparent")

        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "object_detection":
            self.object_detection_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.object_detection_frame.grid_forget()
        if name == "filters_frame":
            self.filters_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.filters_frame.grid_forget()

    def home_frame_btn_event(self):
        """
        Event handler for the home frame button.
        """
        self.select_frame_by_name("home")

    def object_detection_frame_btn_event(self):
        """
        Event handler for the object detection frame button.
        """
        self.current_filter = None
        self.filters_menu.set("Select Filter")
        if self.image_path: 
            self.load_image()
        self.select_frame_by_name("object_detection")

    def filters_frame_btn_event(self):
        """
        Event handler for the filters frame button.
        """
        self.detect_objects_flag = None
        self.object_detection_btn.configure(text='Run Yolo')
        if self.image_path: 
            self.load_image()
        self.select_frame_by_name("filters_frame")

    def change_appearance_mode_event(self, new_appearance_mode):
        """
        Event handler for changing the appearance mode.

        Args:
            new_appearance_mode (str): The new appearance mode to set.
        """
        if new_appearance_mode == "Light Theme": 
            customtkinter.set_appearance_mode('light')
        elif new_appearance_mode == "Dark Theme":
            customtkinter.set_appearance_mode('dark')
        else: 
            customtkinter.set_appearance_mode('system')

if __name__ == "__main__":
    app = FusionFrame()
    app.mainloop()
    
    pid = os.getpid()
    system = psutil.Process(pid)
    system.terminate()
