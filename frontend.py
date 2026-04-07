import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
from ultralytics import YOLO
import queue

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color="#3b82f6", hover_color="#2563eb",
                 fg_color="white", width=220, height=48, radius=12, font_size=11):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.current_color = bg_color
        self.text = text
        self.radius = radius
        self.enabled = True
        self.font_size = font_size

        self.draw()
        self.bind("<Enter>", self.hover)
        self.bind("<Leave>", self.leave)
        self.bind("<Button-1>", self.click)

    def draw(self):
        self.delete("all")
        self.create_rectangle(5, 5, 215, 45, fill="#000000", outline="")
        self.create_rectangle(2, 2, 212, 42, fill=self.current_color, outline="")
        self.create_text(110, 22, text=self.text, fill="white",
                         font=("Segoe UI", self.font_size, "bold"))

    def hover(self, e):
        if self.enabled:
            self.current_color = self.hover_color
            self.draw()

    def leave(self, e):
        if self.enabled:
            self.current_color = self.bg_color
            self.draw()

    def click(self, e):
        if self.enabled and self.command:
            self.command()

    def set_state(self, enabled):
        self.enabled = enabled
        self.current_color = self.bg_color if enabled else "#4b5563"
        self.draw()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Detection Model")
        self.root.geometry("1400x850")

        self.colors = {
            "bg": "#0f172a",
            "card": "#111827",
            "accent": "#3b82f6",
            "green": "#10b981",
            "red": "#ef4444",
            "text": "white",
            "sub": "#9ca3af"
        }

        self.root.configure(bg=self.colors["bg"])

        self.model = YOLO("best.pt")

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False

        self.build_ui()
        self.loop()

    def build_ui(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.topbar()
        self.sidebar()
        self.main_area()

    def topbar(self):
        top = tk.Frame(self.root, bg="#020617", height=65)
        top.grid(row=0, column=0, columnspan=2, sticky="nsew")
        top.grid_propagate(False)

        left = tk.Frame(top, bg="#020617")
        left.pack(side="left", padx=15)

        try:
            logo_img = Image.open("logo.jpg")
            logo_img = logo_img.resize((40, 40))
            self.logo = ImageTk.PhotoImage(logo_img)

            tk.Label(left, image=self.logo, bg="#020617").pack(side="left", padx=(0, 10))
        except:
            pass

        text_frame = tk.Frame(left, bg="#020617")
        text_frame.pack(side="left")

        tk.Label(text_frame,
                 text="Helmet Detection Model",
                 font=("Segoe UI", 16, "bold"),
                 fg="#3b82f6",
                 bg="#020617").pack(anchor="w")

        tk.Label(text_frame,
                 text="AI Safety Monitoring System",
                 font=("Segoe UI", 9),
                 fg="#9ca3af",
                 bg="#020617").pack(anchor="w")

        self.status = tk.Label(top,
                               text="● Ready",
                               bg="#020617",
                               fg="#9ca3af",
                               font=("Segoe UI", 10))
        self.status.pack(side="right", padx=20)

    def sidebar(self):
        side = tk.Frame(self.root, bg=self.colors["card"], width=300)
        side.grid(row=1, column=0, sticky="nsew")

        tk.Label(side, text="Controls", bg=self.colors["card"], fg="white",
                 font=("Segoe UI", 14, "bold")).pack(pady=20)

        ModernButton(side, "📷 Detect Image", self.detect_image).pack(pady=8)
        ModernButton(side, "🎬 Process Video", self.detect_video).pack(pady=8)
        ModernButton(side, "▶ Start Webcam", self.start_webcam,
                     bg_color=self.colors["green"], hover_color="#059669").pack(pady=8)
        ModernButton(side, "⏹ Stop", self.stop_webcam,
                     bg_color=self.colors["red"], hover_color="#dc2626").pack(pady=8)

    def main_area(self):
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.grid(row=1, column=1, sticky="nsew")

        self.canvas = tk.Canvas(main, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=20, pady=20)

    def detect_image(self):
        path = filedialog.askopenfilename()
        if not path: return

        self.update_status("Processing Image...", "#3b82f6")
        res = self.model(path)[0]
        img = res.plot()
        self.show(img)
        self.update_status("Completed", "#10b981")

    def detect_video(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.running = True
        self.update_status("Processing Video...", "#3b82f6")
        threading.Thread(target=self.video_loop, args=(path,), daemon=True).start()

    def video_loop(self, path):
        cap = cv2.VideoCapture(path)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = self.model(frame)[0]
            img = res.plot()
            if not self.frame_queue.full():
                self.frame_queue.put(img)
        cap.release()
        self.update_status("Completed", "#10b981")

    def start_webcam(self):
        self.running = True
        self.update_status("Live Webcam", "#10b981")
        threading.Thread(target=self.webcam_loop, daemon=True).start()

    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            res = self.model(frame)[0]
            img = res.plot()
            if not self.frame_queue.full():
                self.frame_queue.put(img)
        cap.release()

    def stop_webcam(self):
        self.running = False
        self.update_status("Stopped", "#ef4444")

    def show(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((900, 500))
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(450, 250, image=tk_img)
        self.canvas.image = tk_img

    def loop(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.show(frame)
        self.root.after(16, self.loop)

    def update_status(self, text, color="#9ca3af"):
        self.status.config(text=f"● {text}", fg=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()