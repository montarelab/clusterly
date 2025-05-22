import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import ml
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
import numpy as np

class IdeaClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clusterly - Semantic Idea Mapper")

        self.input_ideas = [
            "jogging", "meditation", "yoga", "backend development",
            "JavaScript development", "learn React", "learn C#",
            "mindfulness", "run a marathon"
        ]

        self.colors = ['red', 'blue', 'green']

        # Top: Plot Frame
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom: Entry and Button
        form_frame = ttk.Frame(self.root)
        form_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.entry = ttk.Entry(form_frame, width=50)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.add_button = ttk.Button(form_frame, text="Add Idea", command=self.add_idea)
        self.add_button.pack(side=tk.LEFT)

        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm") 

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
        ml.ml_pipeline(self.ax, self.nlp, self.stop_words, self.input_ideas)
        self.canvas.draw()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = IdeaClusteringApp(root)
    root.mainloop()
