import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import random

from ImageArea import ImageArea, Connection
from detectors import SIFTDetector


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
        self.draw = ImageDraw.Draw(self.img0)

        self.img_tk = ImageTk.PhotoImage(image=self.img0)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Define areas for camera feed (left 2/3) and capture sections (right 1/3)
        self.left_area = ImageArea(0, 0, self.screen_width * 2 / 3, self.screen_height)
        self.areas = self.create_areas()

        # Initialize image holder for captured frames
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
        ret, self.frame_original = self.cap.read()
        if ret:
            # Convert frame to RGB
            self.frame = cv2.cvtColor(self.frame_original, cv2.COLOR_BGR2RGB)
            # Resize frame to fit the left area
            self.frame = cv2.resize(self.frame, (int(self.left_area.x1 - self.left_area.x0),
                                     int(self.left_area.y1 - self.left_area.y0)))
            # Convert to PIL Image
            img_feed = Image.fromarray(self.frame)

            #fill blask
            width, height = self.img0.size
            self.draw.rectangle([(0, 0), (width, height)], fill='black')
            # Draw the camera feed on the left side of img0
            self.img0.paste(img_feed, (int(self.left_area.x0), int(self.left_area.y0)))

            self.update_area_images()
            self.draw_image_connections()

            # Update the canvas image
            self.img_tk = ImageTk.PhotoImage(image=self.img0)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            self.master.after(1, self.update_camera_feed)

            self.left_area.save_image(self.frame, self.frame_original)

    def key_event(self, event):
        print(f"Key pressed: {event.keysym}")
        if event.keysym in ['1', '2', '3', '4']:
            index = int(event.keysym) - 1  # Convert key to index (0-3)
            self.capture_current_frame(index)  # Capture frame when '1', '2', '3', or '4' is pressed

    def capture_current_frame(self, index):
        ret, self.frame_original = self.cap.read()
        if ret:
            # Convert frame to RGB
            self.frame = cv2.cvtColor(self.frame_original, cv2.COLOR_BGR2RGB)

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
            area.save_image(img_captured, self.frame_original)

            self.paste_area_image(area)
            
            self.draw_image_connections()

    def paste_area_image(self, area: ImageArea):
        # Calculate position to center the image in the area
        area.x_offset = int(area.x0 + (area.area_width() - area.new_width) / 2)
        area.y_offset = int(area.y0 + (area.area_height() - area.new_height) / 2)

        # Draw the captured frame in the corresponding rectangle
        self.img0.paste(area.image, (area.x_offset, area.y_offset))

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

    def draw_image_connections(self):
        for i in range(self.max_areas):
            area = self.areas[i]
            if area.image is None:
                continue
            if self.left_area is not None and self.left_area.image is not None:
                self.update_connections(self.left_area, area)

    def update_connections(self, area1: ImageArea, area2: ImageArea) -> None:
        # Ensure both areas have images
        if area1.image is None or area2.image is None:
            return

        det = SIFTDetector()
        result = det.match_features(np.array(area1.image), np.array(area2.image))
        lines = [[(m.point1[0] + area1.x_offset, m.point1[1] + area1.y_offset),
                  (m.point2[0] + area2.x_offset, m.point2[1] + area2.y_offset)]
                 for m in result]

        # Assuming self.img0 is your PIL Image object
        draw = ImageDraw.Draw(self.img0)

        for line in lines:
            # Extract the start and end points of each line
            start_point = line[0]  # (x1, y1)
            end_point = line[1]  # (x2, y2)

            # Draw the line on the image
            draw.line([start_point, end_point],
                      fill="red",  # You can use any color, e.g., (255, 0, 0) for red
                      width=2)  # Line thickness


        print(len(result))
        #print(result)
        #print(area1)
        #print(area2)
        #print(self.img0.height)

        # Generate random points within area1
        points_area1 = [
            (random.randint(area1.x0, area1.x1), random.randint(area1.y0, area1.y1))
            for _ in range(3)
        ]

        # Generate random points within area2
        points_area2 = [
            (random.randint(area2.x0, area2.x1), random.randint(area2.y0, area2.y1))
            for _ in range(3)
        ]

        self.connections = []
        # Create connections between points from area1 and area2
        for i in range(len(points_area1)):
            self.connections.append(Connection(points_area1[i], points_area2[i]))
