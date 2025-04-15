import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk  # Import Pillow for image conversion

class FullScreenApp:
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

        # Create canvas
        self.canvas = tk.Canvas(master, width=self.screen_width, height=self.screen_height)
        self.canvas.pack()

        # Create squares
        self.create_squares()

        # Start camera feed
        self.capture_camera()

    def create_squares(self):
        # Large square (w0)
        self.w0 = self.canvas.create_rectangle(0, 0, self.screen_width * 2 / 3, self.screen_height, fill="blue")

        # Small squares (w)
        self.w = []
        for i in range(4):
            x0 = self.screen_width * 2 / 3
            y0 = (i * self.screen_height / 4)
            x1 = self.screen_width
            y1 = ((i + 1) * self.screen_height / 4)
            square = self.canvas.create_rectangle(x0, y0, x1, y1, fill="red")
            self.w.append(square)

    def capture_camera(self):
        # Capture from the default camera
        self.cap = cv2.VideoCapture(0)
        self.update_camera_feed()

    def update_camera_feed2(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to fit w0
            frame = cv2.resize(frame, (int(self.screen_width * 2 / 3), self.screen_height))
            # Convert to PhotoImage
            img = tk.PhotoImage(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.master.after(10, self.update_camera_feed)
    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to fit w0
            frame = cv2.resize(frame, (int(self.screen_width * 2 / 3), self.screen_height))
            # Convert to PIL Image
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

            self.master.after(10, self.update_camera_feed)
    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)

    def key_event(self, event):
        print(f"Key pressed: {event.keysym}")
        if event.keysym == '1':
            self.capture_current_frame()  # Capture frame when '1' is pressed

    def capture_current_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to fit w0
            frame = cv2.resize(frame, (int(self.screen_width * 2 / 3), self.screen_height))
            # Convert to PIL Image
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Display the captured frame in w0 area
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

    def mouse_event(self, event):
        print(f"Mouse clicked at: {event.x}, {event.y}")

    def exit_app(self, event=None):
        self.cap.release()  # Release the camera
        self.master.destroy()  # Close the application

    def __del__(self):
        self.cap.release()  # Release the camera when the app is closed

if __name__ == "__main__":
    root = tk.Tk()
    app = FullScreenApp(root)
    root.mainloop()
