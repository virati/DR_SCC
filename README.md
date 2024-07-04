# Depression Readout from Subcallosal Cingulate (DR-SCC)
### Dissertation Version

## Overview
The goal of this project was to link oscillations in bilateral subcallosal cingulate cortex (SCC) to depression severity measured with the Hamilton Depression Rating Scale (HDRS).
The resulting decoding model(s) yielded readouts, with the main one being the _depression readout from subcallosal cingulate cortex_ (DR-SCC).

## Methods (brief)
We measured dLFP from six TRD patients implanted with SCCwm-DBS over seven months.
Oscillations calculated from bilateral SCC were then used to _decode_ depression - or find a model that correlatively-links SCC oscillatory power with depression severity 
Oscillations measured over months in the subcallosal cingulate may help us _decode_ depression - or find a measure that can help us better track the depression symptoms of patients.

This repo includes the code for Chapter 3 of my dissertation.
It is also the code for the preprint/pub []().

## Requirements
* ```dbspace``` - Available on PyPi

## Structure
* ```.devcontainer``` contains the meta-information to run this project inside a consistent environment. Requires ```vscode``` and ```vscode::Remote Containers``` plugins; as well as ```Docker```
* ```.dvc``` directory containing meta-info for anonymized + intermediate data cloud storage
* ```analysis``` directory of legacy analyses. No guarantee that these scripts work
* ```notebooks``` the main directory of analyses corresponding to published figures.

## Dissertation
This project was Aim 2/Chapter 3 of my dissertation.
