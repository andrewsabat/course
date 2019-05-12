import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import cv2
import csv
import string
import sys
import os
from sklearn.preprocessing import LabelBinarizer
from PIL import ImageFilter, Image
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from PyQt5.QtGui import QImage, QPainter, QPen, QIcon, QBrush, QPixmap
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtWidgets import *


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.symbol_list = list(string.ascii_letters)

        self.setWindowTitle('mini-Paint')
        self.setWindowIcon(QIcon('images/paint-brush.png'))

        self.setGeometry(800, 400, 400, 400)
        self.setFixedSize(400, 400)

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 30
        self.brushColor = Qt.black

        self.lastPoint = QPoint()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        brushMenu = mainMenu.addMenu('Brush Size')
        brushColor = mainMenu.addMenu('Brush Color')

        saveAction = QAction(QIcon('images/disk.png'), 'Save', self)
        saveAction.setShortcut('Ctrl+S')
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        databaseAction = QAction(QIcon('images/edit.png'), 'Add to Database', self)
        databaseAction.setShortcut('Ctrl+A')
        fileMenu.addAction(databaseAction)
        databaseAction.triggered.connect(self.database)

        recognizeAction = QAction(QIcon('images/magnifier.png'), 'Recognize', self)
        recognizeAction.setShortcut('Ctrl+R')
        fileMenu.addAction(recognizeAction)
        recognizeAction.triggered.connect(self.recognize)

        trainAction = QAction(QIcon('images/bulb.png'), 'Train', self)
        trainAction.setShortcut('Ctrl+T')
        fileMenu.addAction(trainAction)
        trainAction.triggered.connect(self.train)

        clearAction = QAction(QIcon('images/eraser.png'), 'Clear', self)
        clearAction.setShortcut('Ctrl+C')
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        exitAction = QAction(QIcon('images/exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+E')
        fileMenu.addAction(exitAction)
        exitAction.triggered.connect(sys.exit)

        tenpxAction = QAction(QIcon('images/10px.png'), '10 px', self)
        tenpxAction.setShortcut('Ctrl+1')
        brushMenu.addAction(tenpxAction)
        tenpxAction.triggered.connect(self.brushPx)

        twentypxAction = QAction(QIcon('images/20px.png'), '20 px', self)
        twentypxAction.setShortcut('Ctrl+2')
        brushMenu.addAction(twentypxAction)
        twentypxAction.triggered.connect(self.brushPx)

        thirtypxAction = QAction(QIcon('images/30px.png'), '30 px', self)
        thirtypxAction.setShortcut('Ctrl+3')
        brushMenu.addAction(thirtypxAction)
        thirtypxAction.triggered.connect(self.brushPx)

        blackAction = QAction(QIcon('images/black.png'), 'Black', self)
        brushColor.addAction(blackAction)
        blackAction.triggered.connect(self.brushCol)

        redAction = QAction(QIcon('images/red.png'), 'Red', self)
        brushColor.addAction(redAction)
        redAction.triggered.connect(self.brushCol)

        greenAction = QAction(QIcon('images/green.png'), 'Green', self)
        brushColor.addAction(greenAction)
        greenAction.triggered.connect(self.brushCol)

        yellowAction = QAction(QIcon('images/yellow.png'), 'Yellow', self)
        brushColor.addAction(yellowAction)
        yellowAction.triggered.connect(self.brushCol)

        self.train_check()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
            if (event.buttons() & Qt.LeftButton) & self.drawing:
                painter = QPainter(self.image)
                painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.lastPoint, event.pos())
                self.lastPoint = event.pos()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def train_check(self):
        try:
            json_file = open('cnn_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights('cnn_model.h5')
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            QMessageBox.information(self, 'Neural Network Check', "You`r neural network is ready!")
        except FileNotFoundError:
            QMessageBox.warning(self, 'Neural Network Check', "Train network before recognize! (Ctrl + T)")

    def save(self):
        filePath, _= QFileDialog.getSaveFileName(self, "Save image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);; ALL Files(*.*)")
        if filePath == '':
            return
        self.image.save(filePath)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def brushPx(self):
        sender = self.sender().text().split(' ')[0]
        self.brushSize = int(sender)

    def brushCol(self):
        convert = {'Black': Qt.black, 'Red': Qt.red, 'Green': Qt.green, 'Yellow': Qt.yellow}
        sender = self.sender().text()
        self.brushColor = convert[sender]

    def database(self):
        self.img_to_list()
        if not self.inputDialog():
            QMessageBox.warning(self, 'Writing Check', "Incorrect symbol!")
            return
        with open('train.csv', 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.img_list)
            QMessageBox.information(self, 'Writing Check', "Writing done!")

    def inputDialog(self):
        text, result = QInputDialog.getText(self, 'Input Dialog', 'Input the symbol:')
##        if text not in self.symbol_list:
##            return False
        if result:
            self.img_list.insert(0, int(text))
            return True
        else:
            return False
        
    def recognize(self):
        self.img_to_list()
        self.img_list = np.reshape(self.img_list, (1, 28, 28, 1))
        pt.imshow(self.img)
        pt.show()
        predict = self.model.predict(self.img_list)
        max_pr = max(predict[0])
        for i in range(len(predict[0])):
            print ('Chance of ' + str(i) + ' is ' + str(predict[0][i]))
        buttonReply = QMessageBox.question(self, 'Sybmol right check',
        ('Your symbol is ' + str(np.where(predict[0] == max(predict[0]))[0][0]) + '!')
                                           )
        if buttonReply == QMessageBox.No:
            self.database()

    def img_to_list(self):
        image_scaled = self.image.scaled(28, 28)
        image_scaled.save('temp.png')
        image = Image.open('temp.png')
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
        image.save('temp.png')
        img = cv2.imread('temp.png')
        os.remove('temp.png')
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.img_list = [255-item for sublist in self.img for item in sublist]

    def train(self):
        print('Train is starting...')
        train_data = pd.read_csv('train.csv').values

        # training data
        xtrain = train_data[0:, 1:]
        xtrain = xtrain.reshape(len(xtrain),28,28,1)
        ytrain = train_data[0:, 0]
        ytrain = LabelBinarizer().fit_transform(ytrain)

        # create model
        self.model = Sequential()
        
        # add model layers
        self.model.add(Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1)))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())

        self.model.add(Dense(units=120, activation='relu'))

        self.model.add(Dense(units=84, activation='relu'))

        self.model.add(Dense(units=10, activation = 'softmax'))

        # saving model
        model_json = self.model.to_json()
        json_file = open('cnn_model.json', 'w')
        json_file.write(model_json)
        json_file.close()        
        
        # compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # training model
        self.model.fit(xtrain, ytrain, batch_size = 200, epochs = 30, verbose = 2)

        # saving weights
        self.model.save_weights('cnn_model.h5')
        
        print('Train was successful!')
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
