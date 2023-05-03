from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random

import pandas as PD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sb
from fileinput import filename
from tkinter import *
import tkinter as tk
import tkinter
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import pymysql
import pandas as pd
import csv
from csv import writer
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename

from keras import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from login import Login


# ------------------------------------------------------------ Main Window -----------------------------------------



def Loginmeth():
    log = Login()


win = Tk()
win.title("Solar Energy Forecasting ")
win.maxsize(width=1100, height=1000)
win.minsize(width=1100, height=1000)
win.configure(bg='#99ddff')
image1 = Image.open("1.jpg")
img = image1.resize((600, 450))

test = ImageTk.PhotoImage(img)

label1 = tk.Label(win,image=test)
label1.image = test

# Position image
label1.place(x=200, y=400)

#image1 = Image.open("3.png")
test = ImageTk.PhotoImage(img)

label1 = tk.Label(win,image=test)
label1.image = test


# Create Canvas
#canvas1 = Canvas(win, width=400, height=400)

#canvas1.pack(fill="both", expand=True)

# Display image
# canvas1.create_image(0, 0, image=bg, anchor="nw")

# heading label
heading = Label(win, text="Solar Energy Forecasting", font='Verdana 20 bold')
heading.place(x=300, y=50)

btn_login = Button(win, text="Login", font='Verdana 10 bold', width="20", command=Loginmeth)
btn_login.place(x=400, y=200)
btn_exit = Button(win, text="Exit", font='Verdana 10 bold', width="20", command=quit)
btn_exit.place(x=400, y=250)

win.mainloop()
