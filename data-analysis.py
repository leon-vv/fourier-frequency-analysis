import sys
import time
from math import pi

# Names are already prefixed with Q
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit
import pyqtgraph as pg
import numpy as np

import interface

try:
    import nidaqmx as dx
except:
    print("Module nidaqmx not imported")

from scipy import signal


# Utility function
def string_to_float(string, default):
    try:
        return float(string)
    except ValueError:
        return default

class SignalData:

    def fourier_analysis(self, freq):
        d = self.signal_data
        N = len(d)
        return 1/N * np.sum([d[n] * np.exp(-2* pi * 1j * freq * n / float(N)) for n in range(0, N)])

    def __init__(self, time_data, signal_data):

        assert len(time_data) == len(signal_data)

        self.time_data = time_data
        
        self.signal_data = signal_data

        self.freqs = 1/len(signal_data)* np.fft.fft(signal_data) # [self.fourier_analysis(n) for n in range(0, len(signal_data))]
    
    def get_time_data(self):
        return self.time_data

    def get_signal_data(self):
        return self.signal_data
    
    def get_frequencies(self):
        return self.freqs
        
    def get_samples_per_second(self):
        difference = self.time_data[1]-self.time_data[0]
        samples_per_second = 1/ difference
        return samples_per_second
        
    def get_endtime(self):
        return self.time_data[-1]
    
    def get_number_of_samples(self):
        return len(self.time_data)
    

class SineData(SignalData):
    def __init__(self, noise_mean = 0, noise_sigma = 0.1, amplitude=1, frequency=1, eindtijd=1):
        amplitude = amplitude
        
        time_data = np.linspace(0, eindtijd,1000)
        signal_data = amplitude*np.sin(time_data*2*np.pi*frequency)
        signal_data += np.random.normal(noise_mean, noise_sigma, len(time_data))
        
        SignalData.__init__(self, time_data, signal_data)
    
class GaussData(SignalData):
    def __init__(self, amplitude, sigma, end_time):
        time_data = np.linspace(0, end_time, 1000)
        signal_data = amplitude * signal.gaussian(1000, sigma)

        SignalData.__init__(self, time_data, signal_data)

class DeltaData(SignalData):
    def __init__(self, amplitude, end_time):
        amplitude = amplitude
        time_data = np.linspace(0, end_time, 1000)
        signal_data = amplitude * signal.unit_impulse(1000, idx='mid')

        SignalData.__init__(self, time_data, signal_data)

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
        self.incoming_signal = None
        self.interface.setupUi(self.window)
        
        # Setup callbacks
        self.interface.pick_file_button.clicked.connect(self.open_file)
        self.interface.source_picker.currentIndexChanged.connect(self.source_changed)
        self.interface.write_button.clicked.connect(self.write_data)
        # Connect all line edits to method source_changed 
        for k in self.interface.__dict__:
            field = self.interface.__dict__[k]
            if isinstance(field, QLineEdit):
                field.editingFinished.connect(self.source_changed)
        
        # Wait for window to close
        self.source_changed()
        self.window.show()
        sys.exit(self.app.exec_())
        
    def write_data(self):
        
        if('dx' in globals()): # Check if module nidaqmx is loaded
            with dx.Task() as writeTask:
            
                writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')
                max_time = self.signal_data.get_endtime()
                rate = self.signal_data.get_samples_per_second()
                samples = self.signal_data.get_number_of_samples()
            
                writeTask.timing.cfg_samp_clk_timing(rate,
                        sample_mode = dx.constants.AcquisitionType.FINITE,
                        samps_per_chan=samples)   

                writeTask.write(self.signal_data.get_signal_data(),
                        auto_start=True)
                
                with dx.Task() as readTask:
                
                    readTask.ai_channels.add_ai_voltage_chan("myDAQ1/ai1",
                            units=dx.constants.VoltageUnits.VOLTS)
                    
                    readTask.timing.cfg_samp_clk_timing(rate,
                            sample_mode = dx.constants.AcquisitionType.FINITE,
                            samps_per_chan=samples)  

                    values = readTask.read(number_of_samples_per_channel=samples, timeout=10)
                    self.incoming_signal = SignalData(self.signal_data.get_time_data(), values)
                
                
                self.reload_view(True)
 
    def source_changed(self):

        i = self.interface
        tof = string_to_float
        source_text = i.source_picker.currentText()

        if source_text == "Sine":
            mean = tof(i.mean_edit.text(), 0)
            deviation = tof(i.deviation_edit.text(), 0.1)
            amplitude = tof(i.amplitude_edit.text(), 1)
            frequency = tof(i.frequency_edit.text(), 5)
            endtime = tof(i.endtime_edit.text(), 3)
            self.signal_data = SineData(mean, deviation, amplitude, frequency, endtime)

            i.controls_stack.setCurrentWidget(i.sine_controls)
        elif source_text == "Gauss":
            amplitude = tof(i.gauss_amplitude.text(), 1)
            sigma = tof(i.gauss_sigma.text(), 100)
            endtime = tof(i.gauss_end_time.text(), 3)
            self.signal_data = GaussData(amplitude, sigma, endtime)

            i.controls_stack.setCurrentWidget(i.gauss_controls)
        elif source_text == "Delta":
            amplitude = tof(i.delta_amplitude.text(), 2)
            endtime = tof(i.delta_end_time.text(), 3)
            self.signal_data = DeltaData(amplitude, endtime)

            i.controls_stack.setCurrentWidget(i.delta_controls)
        else:
            i.controls_stack.setCurrentWidget(i.file_controls)
         
        self.reload_view(True)
    
    def reload_view(self, auto_range=False):
        
        if self.signal_data == None:
            return
        
        self.interface.generated_time_plot.clear()
        self.interface.generated_frequency_plot.clear()
        self.interface.incoming_time_plot.clear()
        self.interface.incoming_frequency_plot.clear()
        
        pen = pg.mkPen(0.3, width=1)

        self.interface.generated_time_plot.plot(
                self.signal_data.time_data,
                self.signal_data.signal_data,
                pen=pen)
        f = self.signal_data.get_frequencies()
        
        
        self.interface.generated_frequency_plot.plot(np.abs(f), pen=pen)

        
        if self.incoming_signal != None:
            f = self.incoming_signal.get_frequencies()
            
            self.interface.incoming_time_plot.plot(
                self.incoming_signal.time_data,
                self.incoming_signal.signal_data, pen=pen)
            
            self.interface.incoming_frequency_plot.plot(np.abs(f), pen = pen)
                    
        if(auto_range):
            self.interface.generated_time_plot.autoRange()
            self.interface.incoming_frequency_plot.autoRange()
            self.interface.incoming_time_plot.autoRange()
            self.interface.generated_frequency_plot.autoRange()
    
    # Called when a new file is opened
    def open_file(self):
        f = QFileDialog.getOpenFileName(filter="*.txt")
        data = np.loadtxt(f[0])

        sample_time_text = self.interface.sample_time_edit.text()
        sample_time = string_to_float(sample_time_text, 1e-3)
        
        time = np.linspace(0, len(data)*sample_time, len(data))
        
        self.signal_data = SignalData(time, data)
        self.reload_view()

UI()
