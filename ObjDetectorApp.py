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

        # Define areas for the right side of the window
        self.areas = self.create_areas()

        # Initialize image holder for captured frames
        self.img = [None] * 4  # Array to hold 4 captured images

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
            frame = cv2.resize(frame, (int(self.screen_width * 2 / 3), self.screen_height))
            # Convert to PIL Image
            img_feed = Image.fromarray(frame)

            # Draw the camera feed on the left side of img0
            self.img0.paste(img_feed, (0, 0))

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
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get the dimensions of the area
            area = self.areas[index]
            area_width = int(area.x1 - area.x0)
            area_height = int(area.y1 - area.y0)

            # Calculate the aspect ratio of the captured frame
            frame_height, frame_width, _ = frame.shape
            frame_aspect_ratio = frame_width / frame_height

            # Calculate new dimensions while maintaining aspect ratio
            if area_width / area_height > frame_aspect_ratio:
                new_width = area_height * frame_aspect_ratio
                new_height = area_height
            else:
                new_width = area_width
                new_height = area_width / frame_aspect_ratio

            # Resize frame to fit the corresponding area while maintaining aspect ratio
            frame = cv2.resize(frame, (int(new_width), int(new_height)))

            # Convert to PIL Image
            img_captured = Image.fromarray(frame)

            # Save the captured frame in the array
            self.img[index] = img_captured

            # Calculate position to center the image in the area
            x_offset = int(area.x0 + (area_width - new_width) / 2)
            y_offset = int(area.y0 + (area_height - new_height) / 2)

            # Draw the captured frame in the corresponding rectangle
            self.img0.paste(img_captured, (x_offset, y_offset))

            # Update the canvas image
            self.img_tk = ImageTk.PhotoImage(image=self.img0)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)

    def mouse_event(self, event):
        print(f"Mouse clicked at: {event.x}, {event.y}")

    def exit_app(self, event=None):
        self.cap.release()  # Release the camera
        self.master.destroy()  # Close the application

    def __del__(self):
        self.cap.release()  # Release the camera when the app is closed


