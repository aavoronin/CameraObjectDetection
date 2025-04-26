import tkinter as tk
import cv2
from PIL import Image, ImageTk

from ImageArea import ImageArea


class ObjDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.attributes('-fullscreen', True)  # Set full screen
        self.master.bind("<Escape>", self.exit_fullscreen)  # Bind escape key to exit fullscreen
        self.master.bind("<Key>", self.key_event)  # Bind key events
        self.master.bind("<Button-1>", self.mouse_event)  # Bind mouse click events
        self.master.bind("<Alt-x>", self.exit_app)  # Bind Alt + X to exit the app

        # Get screen size
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()

        # Create a canvas
        self.canvas = tk.Canvas(master, width=self.screen_width, height=self.screen_height)
        self.canvas.pack()

        # Create a single image that covers the whole window
        self.img0 = Image.new("RGB", (self.screen_width, self.screen_height), (0, 0, 0))
        self.img_tk = ImageTk.PhotoImage(image=self.img0)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Define areas for camera feed (left 2/3) and capture sections (right 1/3)
        self.left_area = ImageArea(0, 0, self.screen_width * 2 / 3, self.screen_height)
        self.areas = self.create_areas()

        ## Initialize image holder for captured frames
        #self.img = [None] * 4  # Array to hold 4 captured images
        self.max_areas = 4

        # Start camera feed
        self.capture_camera()

    def create_areas(self):
        # Create a list to hold the area structures
        areas = []
        x0 = self.screen_width * 2 / 3
        for i in range(4):
            y0 = (i * self.screen_height / 4)
            x1 = self.screen_width
            y1 = ((i + 1) * self.screen_height / 4)
            areas.append(ImageArea(x0, y0, x1, y1))
        return areas

    def capture_camera(self):
        # Capture from the default camera
        self.cap = cv2.VideoCapture(0)
        self.update_camera_feed()

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to fit the left area
            frame = cv2.resize(frame, (int(self.left_area.x1 - self.left_area.x0),
                                     int(self.left_area.y1 - self.left_area.y0)))
            # Convert to PIL Image
            img_feed = Image.fromarray(frame)

            # Draw the camera feed on the left side of img0
            self.img0.paste(img_feed, (int(self.left_area.x0), int(self.left_area.y0)))

            self.update_area_images()

            # Update the canvas image
            self.img_tk = ImageTk.PhotoImage(image=self.img0)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            self.master.after(1, self.update_camera_feed)

    def key_event(self, event):
        print(f"Key pressed: {event.keysym}")
        if event.keysym in ['1', '2', '3', '4']:
            index = int(event.keysym) - 1  # Convert key to index (0-3)
            self.capture_current_frame(index)  # Capture frame when '1', '2', '3', or '4' is pressed

    def capture_current_frame(self, index):
        ret, self.frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Get the dimensions of the area
            area = self.areas[index]

            # Calculate the aspect ratio of the captured frame
            self.frame_height, self.frame_width, _ = self.frame.shape
            self.frame_aspect_ratio = self.frame_width / self.frame_height

            # Calculate new dimensions while maintaining aspect ratio
            if area.area_width() / area.area_height() > self.frame_aspect_ratio:
                area.new_width = area.area_height() * self.frame_aspect_ratio
                area.new_height = area.area_height()
            else:
                area.new_width = area.area_width()
                area.new_height = area.area_width() / self.frame_aspect_ratio

            # Resize frame to fit the corresponding area while maintaining aspect ratio
            self.frame = cv2.resize(self.frame, (int(area.new_width), int(area.new_height)))

            # Convert to PIL Image
            img_captured = Image.fromarray(self.frame)
            # Save the captured frame in the array
            area.save_image(img_captured)

            self.paste_area_image(area)

    def paste_area_image(self, area):
        # Calculate position to center the image in the area
        x_offset = int(area.x0 + (area.area_width() - area.new_width) / 2)
        y_offset = int(area.y0 + (area.area_height() - area.new_height) / 2)
        # Draw the captured frame in the corresponding rectangle
        self.img0.paste(area.image, (x_offset, y_offset))

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)

    def mouse_event(self, event):
        print(f"Mouse clicked at: {event.x}, {event.y}")

    def exit_app(self, event=None):
        self.cap.release()  # Release the camera
        self.master.destroy()  # Close the application

    def __del__(self):
        self.cap.release()  # Release the camera when the app is closed

    def update_area_images(self):
        for i in range(self.max_areas):
            area = self.areas[i]
            if area.image is None:
                continue
            self.paste_area_image(area)

        # Update the canvas image
        # self.img_tk = ImageTk.PhotoImage(image=self.img0)
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
