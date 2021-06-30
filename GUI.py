import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets, uic
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap
from read_file import *
from Quality import quality
class Main_Window(QMainWindow):

    def __init__(self):
        """In def__init__(self) part Qt designer .ui file is loaded,
        all tools are defined here used in Qt designer
        also their values are defined here"""
        super().__init__()  # call super function for override
        self.main_window = uic.loadUi("GUI.ui", self)  # load ui file
        # iteration_slider = QSlider
        self.show()

        self.main_window.refresh_button.clicked.connect(self.refresh_function)
        self.main_window.time_spin.valueChanged.connect(self.time_spinner)
        self.main_window.time_spin.setValue(10)
        self.time_value = self.main_window.time_spin.value()
        self.main_window.comboBox.currentIndexChanged.connect(self.getComboValue)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.canvas)
        self.main_window.label.setLayout(self.layout)
        self.main_window.setLayout(self.layout)
        pixmap = QPixmap("logo.png")  # IKC logo
        self.label_logo.setPixmap(QPixmap(pixmap))
        # load image to label
        self.resize(pixmap.width(), pixmap.height())


    def temperature_function(self):

        X_test, y_test, y_pred = self.machine_learning()

        mymodel3 = np.poly1d(np.polyfit(X_test[:, 2], y_test[:, 0], 3))
        myline3 = np.linspace(25, 32, 100)

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.scatter(X_test[:, 2], y_test[:, 0], color='black')
        plt.plot(myline3, mymodel3(myline3), color='black')
        plt.ylim([50, 100])
        ax.set_xlabel('Temperature (C)')
        ax.set_ylabel('Quality (%)')
        ax.set_title('Temperature vs Quality')

        rscore = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        self.main_window.label_4.setText(str('Best result at'))
        self.main_window.label_3.setText(str('25 C'))
        self.main_window.label_6.setText(str('R-squared value is'))
        self.main_window.label_6.setStyleSheet("color:black")
        self.main_window.label_7.setText(str(format(rscore, ".4f")))
        self.main_window.label_8.setText(str('MSE value is'))
        self.main_window.label_9.setText(str(format(mse, ".4f")))

        self.canvas.draw()

    def humidity_function(self):

        X_test, y_test, y_pred = self.machine_learning()

        mymodel = np.poly1d(np.polyfit(X_test[:, 0], y_test[:, 0], 3))
        myline = np.linspace(33, 70, 100)

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.scatter(X_test[:, 0], y_test[:, 0], color='blue')
        plt.plot(myline, mymodel(myline), color='blue')
        plt.ylim([50, 100])
        ax.set_xlabel('Humidity (%)')
        ax.set_ylabel('Quality (%)')
        ax.set_title('Humidity vs Quality')

        rscore = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        self.main_window.label_4.setText(str('Best result at'))
        self.main_window.label_3.setText(str('50%'))
        self.main_window.label_6.setText(str('R-squared value is'))
        self.main_window.label_6.setStyleSheet("color:black")
        self.main_window.label_7.setText(str(format(rscore, ".4f")))
        self.main_window.label_8.setText(str('MSE value is'))
        self.main_window.label_9.setText(str(format(mse, ".4f")))

        self.canvas.draw()

    def pressure_function(self):

        X_test, y_test, y_pred = self.machine_learning()

        mymodel2 = np.poly1d(np.polyfit(X_test[:, 1], y_test[:, 0], 3))
        myline2 = np.linspace(1005, 1010, 100)

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.scatter(X_test[:, 1], y_test[:, 0], color='red')
        plt.plot(myline2, mymodel2(myline2), color='red')
        plt.ylim([50, 100])
        ax.set_xlabel('Air pressure (hPa)')
        ax.set_ylabel('Quality (%)')
        ax.set_title('Air pressure vs Quality')

        rscore = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        self.main_window.label_3.setText(str())
        self.main_window.label_4.setText(str())
        self.main_window.label_6.setText(str('R-squared value is'))
        self.main_window.label_6.setStyleSheet("color:black")
        self.main_window.label_7.setText(str(format(rscore, ".4f")))
        self.main_window.label_8.setText(str('MSE value is'))
        self.main_window.label_9.setText(str(format(mse, ".4f")))

        self.canvas.draw()

    def temperature_time(self):

        sensordata = pd.read_csv("log_q.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 2], color='black', label='Real temperature line')
        plt.axhline(y=25, color='g', label='Referance line')
        plt.ylim([0, 50])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature vs Time')

        sum = 0
        for i in inputs[-2 * x:, 2]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Temperature:'))
        if 28 >= average >= 23:
            self.main_window.label_6.setText(str('Normal Temperature.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 28:
            self.main_window.label_6.setText(str('High Temperature!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 23:
            self.main_window.label_6.setText(str('Low Temperature!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())

        self.canvas.draw()

    def temperature_time_2(self):

        sensordata = pd.read_csv("logfridge.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 2], color='black', label='Real temperature line')
        plt.axhline(y=8, color='g', label='Referance line')
        plt.ylim([0, 50])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature vs Time')

        sum = 0
        for i in inputs[-2 * x:, 2]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Temperature:'))
        if 10 >= average >= 6:
            self.main_window.label_6.setText(str('Normal Temperature.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 10:
            self.main_window.label_6.setText(str('High Temperature!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 6:
            self.main_window.label_6.setText(str('Low Temperature!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())

        self.canvas.draw()

    def humidity_time(self):

        sensordata = pd.read_csv("log_q.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 0], color='blue', label='Real humidity line')
        plt.axhline(y=50, color='g', label='Referance line')
        plt.ylim([0, 100])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Humidity')
        ax.set_title('Humidity vs Time')

        sum = 0
        for i in inputs[-2 * x:, 0]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Humidity:'))
        if 60 >= average >= 40:
            self.main_window.label_6.setText(str('Normal Humidity.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 60:
            self.main_window.label_6.setText(str('High Humidity!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 40:
            self.main_window.label_6.setText(str('Low Humidity!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())

        self.canvas.draw()

    def humidity_time_2(self):

        sensordata = pd.read_csv("logfridge.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 0], color='blue', label='Real humidity line')
        plt.axhline(y=60, color='g', label='Referance line')
        plt.ylim([0, 100])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Humidity')
        ax.set_title('Humidity vs time')

        sum = 0
        for i in inputs[-2 * x:, 0]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Humidity:'))
        if 70 >= average >= 50:
            self.main_window.label_6.setText(str('Normal Humidity.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 70:
            self.main_window.label_6.setText(str('High Humidity!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 50:
            self.main_window.label_6.setText(str('Low Humidity!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())

        self.canvas.draw()

    def pressure_time(self):

        sensordata = pd.read_csv("log_q.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 1], color='red', label='Real air pressure line')
        plt.axhline(y=1000, color='g', label='Referance line')
        plt.ylim([950, 1050])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Air pressure (hPa)')
        ax.set_title('Air pressure (hPa) vs time')

        sum = 0
        for i in inputs[-2 * x:, 1]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Air Pressure:'))
        if 1020 >= average >= 980:
            self.main_window.label_6.setText(str('Normal Air Pressure.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 1020:
            self.main_window.label_6.setText(str('High Air Pressure!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 980:
            self.main_window.label_6.setText(str('Low Air Pressure!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())

        self.canvas.draw()

    def pressure_time_2(self):

        sensordata = pd.read_csv("logfridge.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 4:6].values

        x = self.time_value

        ax = self.figure.add_subplot(111)
        ax.clear()
        plt.plot(output[-2 * x:, 1], inputs[-2 * x:, 1], color='red', label='Real air pressure line')
        plt.axhline(y=1000, color='g', label='Referance line')
        plt.ylim([950, 1050])
        plt.legend(loc="upper left", borderaxespad=0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Air pressure (hPa)')
        ax.set_title('Air pressure (hPa) vs time')

        sum = 0
        for i in inputs[-2 * x:, 1]:
            sum += i
        average = sum/(2*x)
        self.main_window.label_3.setText(str(format(average, ".4f")))
        self.main_window.label_4.setText(str('Average Air Pressure:'))
        if 1020 >= average >= 980:
            self.main_window.label_6.setText(str('Normal Air Pressure.'))
            self.main_window.label_6.setStyleSheet("color:green")
        if average > 1020:
            self.main_window.label_6.setText(str('High Air Pressure!'))
            self.main_window.label_6.setStyleSheet("color:red")
        if average < 980:
            self.main_window.label_6.setText(str('Low Air Pressure!'))
            self.main_window.label_6.setStyleSheet("color:blue")
        self.main_window.label_7.setText(str())
        self.main_window.label_8.setText(str())
        self.main_window.label_9.setText(str())
        
        self.canvas.draw()

    def getComboValue(self):
        """when types of image is selected
        from combo box, take their names with converter value"""
        if self.main_window.comboBox.currentText() == "Office Regression":
            self.main_window.temperature_button.clicked.connect(self.temperature_function)
            self.main_window.humidity_button.clicked.connect(self.humidity_function)
            self.main_window.pressure_button.clicked.connect(self.pressure_function)

        if self.main_window.comboBox.currentText() == "Office Time":
            self.main_window.temperature_button.clicked.connect(self.temperature_time)
            self.main_window.humidity_button.clicked.connect(self.humidity_time)
            self.main_window.pressure_button.clicked.connect(self.pressure_time)

        if self.main_window.comboBox.currentText() == "Fridge Time":
            self.main_window.temperature_button.clicked.connect(self.temperature_time_2)
            self.main_window.humidity_button.clicked.connect(self.humidity_time_2)
            self.main_window.pressure_button.clicked.connect(self.pressure_time_2)


    def machine_learning(self):
        sensordata = pd.read_csv("log_q.csv")
        inputs = sensordata.iloc[:, 0:3].values
        output = sensordata.iloc[:, 6:7].values

        X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2)
        regr = ExtraTreesRegressor()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        return X_test,y_test,y_pred


    def refresh_function(self):
        read_file()
        quality()

    def time_spinner(self):
        self.time_value = self.main_window.time_spin.value()


def run():
    app = QApplication(sys.argv)  # call Application
    window = Main_Window()  # call window class
    sys.exit(app.exec())  # enter infinity loop

if __name__ == "__main__":  # run if only script is main
    run()  # call run function




