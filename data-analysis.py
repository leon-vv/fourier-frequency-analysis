import sys
import time
import math

# Names are already prefixed with Q
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit
import pyqtgraph as pg
import numpy as np

import interface
from scipy import signal, optimize


try:
    import nidaqmx as dx
except:
    print("Module nidaqmx not imported")


# Utility functions
def string_to_float(string, default):
    try:
        return float(string)
    except ValueError:
        return default

def mydaq_loaded():
    return 'dx' in globals()

class SignalData:

    def fourier_analysis(self, freq):
        d = self.signal_data
        N = len(d)
        return 1/N * np.sum([d[n] * np.exp(-2* pi * 1j * freq * n / float(N)) for n in range(0, N)])

    def __init__(self, time_data, signal_data):
        
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
    
    def mydaq_response(self):
   
        if(not mydaq_loaded()): return

        with dx.Task() as writeTask:
            with dx.Task() as readTask:
                writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')
                
                rate = self.get_samples_per_second()
                
                N = len(self.signal_data)
                
                writeTask.timing.cfg_samp_clk_timing(rate,
                        sample_mode = dx.constants.AcquisitionType.FINITE,
                        samps_per_chan=N)
                
                readTask.ai_channels.add_ai_voltage_chan("myDAQ1/ai1",
                        units=dx.constants.VoltageUnits.VOLTS)
                
                readTask.timing.cfg_samp_clk_timing(rate,
                        sample_mode = dx.constants.AcquisitionType.CONTINUOUS)
                
                readTask.start()
                
                writeTask.write(self.signal_data, auto_start=True)
                
                # To make sure we read all the samples written, read half a second longer
                extra_samples_to_read = math.ceil(500e-3 * rate) 

                response = readTask.read(
                        number_of_samples_per_channel= N + extra_samples_to_read,
                        timeout=dx.constants.WAIT_INFINITELY)
                
                # First index bigger than 0.1
                # Returns 0 if no value is larger than 0.1 volts
                start_index = np.argmax(np.abs(response) > 0.1) 
                
                response = response[start_index:start_index + N]
                
                return SignalData(np.linspace(0, N/rate, N), response)
 
    def get_amplitude_and_phase(self, frequency):
    
        fit_function = lambda t, A, phase: A * np.sin(2*np.pi*frequency*t + phase)
         
        parameters, _ = optimize.curve_fit(fit_function,
                self.time_data,
                self.signal_data,
                bounds=([0, 0], [np.inf, np.pi]))
        
        return parameters
 

class SineData(SignalData):
    def __init__(self, noise_mean = 0, noise_sigma = 0.1, amplitude=1, frequency=1, eindtijd=1):
        amplitude = amplitude
        
        time_data = np.linspace(0, eindtijd, 2000)
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
        
        # Set plot labels
        
        self.interface.generated_time_plot.setLabel('left', 'Voltage', 'V')
        self.interface.generated_time_plot.setLabel('bottom', 'time', 's')
        self.interface.generated_frequency_plot.setLabel('left', 'Amplitude')
        self.interface.generated_frequency_plot.setLabel('bottom', 'frequency', 'Hz')
        self.interface.incoming_time_plot.setLabel('left', 'Voltage', 'V')
        self.interface.incoming_time_plot.setLabel('bottom', 'time', 's')
        self.interface.incoming_frequency_plot.setLabel('left', 'Amplitude')
        self.interface.incoming_frequency_plot.setLabel('bottom', 'frequency', 'Hz')
        self.interface.amplitude_plot.setLabel('left', 'Amplitude')
        self.interface.amplitude_plot.setLabel('bottom', 'Frequency', 'Hz')
        self.interface.phase_plot.setLabel('left', 'Phase', 'rad')
        self.interface.phase_plot.setLabel('bottom', 'Frequency', 'Hz')
        

        # Setup event callbacks
        self.interface.pick_file_button.clicked.connect(self.open_file)
        self.interface.source_picker.currentIndexChanged.connect(self.source_changed)
        self.interface.write_button.clicked.connect(self.write_pe3_data)
        self.interface.bode_plot_button.clicked.connect(self.make_bode_plot)

        # Connect all line edits to method source_changed 
        for k in self.interface.__dict__:
            field = self.interface.__dict__[k]
            if isinstance(field, QLineEdit):
                field.editingFinished.connect(self.source_changed)
        
        # Wait for window to close
        self.source_changed()
        self.window.show()
        sys.exit(self.app.exec_())
    
    def write_pe3_data(self):
     
        self.incoming_signal = self.signal_data.mydaq_response()
        self.reload_pe3_view(True)
    
    def make_bode_plot(self):
    
        if(not mydaq_loaded()): return
        
        self.interface.amplitude_plot.clear()
        self.interface.phase_plot.clear()
        
        tof = string_to_float
        freq_start = tof(self.interface.frequency_start_edit.text(), 10)
        freq_end = tof(self.interface.frequency_end_edit.text(), 400)

        freqs = np.linspace(freq_start, freq_end, 25)
        input_amplitude = 5
        amplitudes = []
        phases = []

        for f in freqs:
        
            self.app.processEvents() # Prevent UI from hanging

            endtime = 5 / f # 5 whole periods of oscillation
            sine = SineData(0, 0, input_amplitude, f, endtime)
            
            (A, phase) = sine.mydaq_response().get_amplitude_and_phase(f)

            amplitudes.append(A)
            phases.append(phase)

        amplitudes = np.array(amplitudes) / input_amplitude

        self.interface.amplitude_plot.plot(freqs, amplitudes, pen=None, symbolSize=6, symbol='o')
        self.interface.phase_plot.plot(freqs, phases, pen=None, symbolSize=6, symbol='o')
    
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
         
        self.reload_pe3_view(True)
    
    def reload_pe3_view(self, auto_range=False):
    
        if self.signal_data == None or self.interface.tabs.currentIndex() != 0:
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
        self.reload_pe3_view()

UI()
