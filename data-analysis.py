import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import pyqtgraph as pg

import interface

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QApplication(sys.argv)

window = QMainWindow() 

generated_interface = interface.Ui_MainWindow()
generated_interface.setupUi(window)
generated_interface.plot_view.plot(range(100), range(100))
print( "hoi")
window.show()
sys.exit(app.exec_())

