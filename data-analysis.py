import sys
from math import pi

# Names are already prefixed with Q
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np
import interface
from scipy import signal

class SignalData:

    def fourier_analysis(self, freq):
        d = self.signal_data
        N = len(d)
        return np.sum([d[n] * np.exp(-2* pi * 1j * freq * n / float(N)) for n in range(0, N)])

    def __init__(self, time_data, signal_data):

        assert len(time_data) == len(signal_data)

        self.time_data = time_data
        self.signal_data = signal_data

        self.freqs = np.fft.fft(signal_data) # [self.fourier_analysis(n) for n in range(0, len(signal_data))]
    
    def get_frequencies(self):
        return self.freqs

class SineData(object):
    def __init__(self, noise_mean = 0, noise_sigma = 0.1, amplitude=1, frequency=1, eindtijd=1):
        self.amplitude = amplitude
        self.frequencies= [frequency]
        self.time_data = np.linspace(0, eindtijd,1000)
        self.signal_data= amplitude*np.sin(self.time_data*2*np.pi*frequency)
        self.signal_data += np.random.normal(noise_mean, noise_sigma, len(self.time_data))
    
    def get_frequencies(self):
        return self.frequencies
    
    
    def gokje(self, x,a, b, c, d):
        return a*np.sin(c*(x+d))+b
  
    def evaluate(self):
            params, parcov = optimize.curve_fit(self.gokje, xdata=self.x, ydata=self.y_data)
            print(params, parcov)
            fit_data = params[0]*np.sin(params[2]*(self.x+params[3]))+params[1]
            plt.plot(self.time_data, fit_data, label = "Fit")
            plt.legend()
            plt.show()

class GaussData(SignalData):
    def __init__(self, mean, sigma, maxtime):
        
        time_data = np.linspace(0, maxtime, 1000)
        signal_data = signal.gaussian(1000, sigma)
        SignalData.__init__(self, time_data, signal_data)


class DeltaData(SignalData):
    def __init__(self, mean, amplitude, maxtime):
        self.mean = mean
        self.amplitude = amplitude
        self.signal_data = amplitude * signal.unit_impulse(1000, round((mean/maxtime)*1000))

        
    
        
            
class UI:

    def __init__(self):
    
         # Config
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.signal_data = None

        # Init window
        self.app = QApplication(sys.argv)
        self.window = QMainWindow() 
        self.interface = interface.Ui_MainWindow()

        self.interface.setupUi(self.window)
        
        # Setup callbacks
        self.interface.pick_file_button.clicked.connect(self.open_file)
        self.interface.domain_picker.currentIndexChanged.connect(lambda: self.reload_view(True))

        self.interface.amplitude_edit.editingFinished.connect(self.source_changed)
        self.interface.frequency_edit.editingFinished.connect(self.source_changed)
        self.interface.mean_edit.editingFinished.connect(self.source_changed)
        self.interface.deviation_edit.editingFinished.connect(self.source_changed)
        self.interface.endtime_edit.editingFinished.connect(self.source_changed)
        
        self.interface.source_picker.currentIndexChanged.connect(self.source_changed)
        
        # Wait for window to close
        self.source_changed()
        self.window.show()
        sys.exit(self.app.exec_())
        
    
    def source_changed(self):
        if self.interface.source_picker.currentText() == "Sine":
            self.interface.sine_controls.setVisible(True)
            mean = float(self.interface.mean_edit.text())
            deviation = float(self.interface.deviation_edit.text())
            amplitude = float(self.interface.amplitude_edit.text())
            frequency = float(self.interface.frequency_edit.text())
            endtime = float(self.interface.endtime_edit.text())
            self.signal_data = SineData(mean, deviation, amplitude, frequency, endtime)
            self.interface.file_control.hide()
        if self.interface.source_picker.currentText() == "File":
            self.interface.file_control.setVisible(True)
            self.interface.sine_controls.hide()
    
        self.reload_view(True)
    
    def reload_view(self, auto_range=False):
        
        if self.signal_data == None:
            return
        
        view = self.interface.domain_picker.currentText() 
        
        self.interface.plot_view.clear()
        pen = pg.mkPen(0.3, width=1)

        if view == "Time Domain":
            self.interface.plot_view.plot(
                    self.signal_data.time_data,
                    self.signal_data.signal_data,
                    pen=pen)
        else:
            f = self.signal_data.get_frequencies()
            self.interface.plot_view.plot(np.abs(f), pen=pen)

        if(auto_range): self.interface.plot_view.autoRange()

    # Called when a new file is opened
    def open_file(self):
        f = QFileDialog.getOpenFileName(filter="*.txt")
        data = np.loadtxt(f[0])

        # Currently hardcoded: 1ms sample rate
        # Improvement: read time data from differenct column, configurable sample rate
        time = np.linspace(0, len(data)*1e-3, len(data))
        
        self.signal_data = SignalData(time, data)
        self.reload_view()

UI()
