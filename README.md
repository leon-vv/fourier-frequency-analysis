# Spectrum Analysis Tool

This is a signal analysis tool which allows the user to acquire an input from different
kind of sources, then apply a filter in the frequency domain and subsequently output the data to a MyDaq USB device or a file.

## Input

Input sources are:
* A file with voltage data
* A MyDaq device connected to the computer
* A Rigol oscilloscope connected to the computer

This input signal can be viewed in the time and frequency domain. The frequency
domain data is generated using a discrete fourier analysis.

## Filter
It's possible to apply a filter to this spectrum. This filter can be applied both
at the frequency level (e.g. remove frequencies 50Hz and 60Hz or keep only frequency 40Hz) or at the amplitude level (e.g. remove all frequencies with an amplitude lower than 30).

## Output
The filtered signal can be viewed again, either in the frequency or time domain. It's also possibly to output the signal again. Supported output targets are
* A file on the computer
* A MyDaq device
