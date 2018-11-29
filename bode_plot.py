from nidaqmx.task import Task 
from nidaqmx import constants 
from scipy.optimize import curve_fit
import numpy as np


class read_writer():
    def __init__(self, freq, write_data, sample_rate, number_of_samples):
        """Function to read and write with the MyDAQ. Write data should be array with length equal to number_of_samples
       Make sure array does not exceed limits of MyDAQ as no safeguards are built into this function"""
        #if len(write_data) != number_of_samples:
        #    print("Wrong number of samples passed. Breaking now")
        #    return 0
        self.freq = freq
        self.__define_tasks__(write_data, sample_rate, number_of_samples)
        self.Nsamples = number_of_samples
        self.rate = sample_rate

    def __define_tasks__(self,write_data, sample_rate, number_of_samples):
        """Here we define the read and write tasks, including their timing"""
        self.readTask = Task()
        # Set channel to read from
        self.readTask.ai_channels.add_ai_voltage_chan(physical_channel = 'myDAQ1/ai0')
        self.readTask.ai_channels.add_ai_voltage_chan(physical_channel = 'myDAQ1/ai1')
        
        # Configure timing
        self.readTask.timing.cfg_samp_clk_timing(rate = sample_rate, samps_per_chan = number_of_samples, sample_mode=constants.AcquisitionType.FINITE)
        
        # We define and configure the write task
        self.writeTask = Task()
        # Set channel
        self.writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao0')
        self.writeTask.ao_channels.add_ao_voltage_chan('myDAQ1/ao1')
        # Set timing
        self.writeTask.timing.cfg_samp_clk_timing(rate = sample_rate, samps_per_chan = number_of_samples, sample_mode=constants.AcquisitionType.FINITE)
        
        
        self.xarr_fit = np.linspace(0, number_of_samples/sample_rate, number_of_samples)
        self.yarr_fit = np.sin(2*np.pi*self.freq*self.xarr_fit)
        #self.fitfreq = (number_of_samples / sample_rate)/5.
        #self.fit_data = 3*np.sin(2*np.pi*self.fitfreq*self.xarr_fit)
        self.writeTask.write([list(self.yarr_fit),list(write_data)])

    def __start__(self):
        """Here we start the writing and reading. We wait until reading is finished and then read the buffer from the DAQ"""
        self.readTask.start()
        self.writeTask.start()

        self.readTask.wait_until_done()
        self.writeTask.wait_until_done()
        # Acquire data
        self.__acquire__()
        self.__close__()
        self.__find_phi__()
    def __close__(self):
        """Here we shut the tasks explicitly. This function is called from the __start__ function"""
        # Close connection
        self.readTask.close()
        self.writeTask.close()

    def __acquire__(self):
        """Read the buffer into attribute self.data"""
        # Retrieve data
        self.input = self.readTask.read(number_of_samples_per_channel = self.Nsamples)

        self.data = self.input[1]
        self.fit_in = self.input[0]

    def __fit_call__(self, x, t0):
        return np.sin(2*np.pi*self.freq*(x-t0))

    def __find_phi__(self):    
        popt,pcov = curve_fit(self.__fit_call__, self.xarr_fit[1000:], self.fit_in[1000:], p0=0, maxfev=int(1e6))
        self.t0 = popt[0]


    def __output__(self):
        """Return a time array and the data"""
        # Prepare return data
        #time_arr = np.linspace(0, self.Nsamples/self.rate, self.Nsamples)
        t_arr = np.linspace(0, self.Nsamples/self.rate, self.Nsamples ) - self.t0
        return t_arr, np.array(self.data), self.t0
    
    
    freqarr = np.logspace(0,3,10)

def get_amplitudes_and_phases(freqarr):
    freqarr = np.logspace(0,3,10)

    duration=3#sec
    sample_rate=90000

    def fitter(x, phi):
        return np.sin(2*np.pi*freq*x + phi)

    t = np.linspace(0, duration, int(sample_rate*duration))

    phases, amps = np.zeros(len(freqarr)),np.zeros(len(freqarr))
    for i, freq in enumerate(freqarr):
        data = np.sin(2*np.pi*freq*t)
        obj = read_writer(freq,data, sample_rate, int(sample_rate*duration))
        obj.__start__()
        t_arr, dataread, t0 = obj.__output__()

        t_shift = int(np.ceil(-t_arr[0]/np.diff(t)[0]))

        tofit = np.where((t_arr>1)*(t_arr<2))[0]
        popt,pcov = curve_fit(fitter, t_arr[tofit], dataread[tofit]/np.max(dataread[tofit]), p0=[0], maxfev=int(1e6))
        recovered_phase_shift = popt[0]

        print("t0: ", np.round(t0,5), " phi: ", np.round(recovered_phase_shift,2), "rad amp: ", np.round(np.max(dataread)/np.max(data),2))
        phases[i], amps[i] = recovered_phase_shift, np.max(dataread)/np.max(data)

    return phases, amps
