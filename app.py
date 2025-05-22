import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import style as mpl_style
import ml
from nltk.corpus import stopwords
import spacy
import numpy as np
from loguru import logger as log

class IdeaClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clusterly - Semantic Idea Mapper")
        self.root.configure(bg="#0f0f1a")  # Deep dark blue background

        # Use dark matplotlib style
        mpl_style.use('dark_background')

        self.input_ideas = [
            "jogging", "meditation", "yoga", "backend development",
            "JavaScript development", "learn React", "learn C#",
            "mindfulness", "run a marathon"
        ]

        log.info('App initial setup')

        self.stop_words = set(stopwords.words('english'))
        log.info('Loaded stop words')

        self.nlp = spacy.load("en_core_web_sm")
        log.info('Loaded language model')


        # Setup dark-themed style for ttk
        style = ttk.Style()
        style.theme_use("clam")  # More customizable
        style.configure("TFrame", background="#0f0f1a")
        style.configure("TLabel", background="#0f0f1a", foreground="#00ffe1")
        style.configure("TButton", background="#1e1e2f", foreground="#00ffe1", font=("DejaVu Sans Mono", 10, "bold"))
        style.map("TButton", background=[("active", "#292940")])
        style.configure("TEntry", fieldbackground="#1e1e2f", foreground="#00ffe1", insertcolor="#00ffe1")

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.patch.set_facecolor('#0f0f1a')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Form frame
        form_frame = ttk.Frame(self.root)
        form_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.entry = ttk.Entry(form_frame, width=50)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.add_button = ttk.Button(form_frame, text="Add Idea", command=self.add_idea)
        self.add_button.pack(side=tk.LEFT)

        # Initial plot
        self.update_plot()

    def add_idea(self):
        new_idea = self.entry.get().strip()
        if new_idea:
            self.input_ideas.append(new_idea)
            self.entry.delete(0, tk.END)
            self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('#0f0f1a')
        ml.ml_pipeline(self.ax, self.nlp, self.stop_words, self.input_ideas)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = IdeaClusteringApp(root)
    root.mainloop()
