#! /usr/local/bin/python3.7

import numpy as np
from PyQt5.QtWidgets import *


class Window(QMainWindow):
	def __init__(self):
		super().__init__()

		self.title = "PyQt5 Window"
		self.top = 100
		self.left = 100
		self.width = 400
		self.height = 300

	def InitWindow(self):
		self.setWindowTitle(self.title)