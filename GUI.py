from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from testtensor import main

class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Handwritten digit recognition")
        self.minsize(250, 150)
        self.wm_iconbitmap('mnist.ico')

        self.labelFrame = ttk.LabelFrame(self, text = "Select the handwritten digit image that needed recognize")
        self.labelFrame.grid(column = 0, row = 0, padx = 125, pady = 75)

        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse File", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir = "testimages/", title = "Select A Image", filetype = (("jpg file", "*.jpg"), ("png file", "*.png"), ("all file", "*.*")))
        main(self.filename)

root = Root()
root.mainloop()