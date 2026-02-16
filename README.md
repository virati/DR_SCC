# Depression Oscillatory-Readout from Subcallosal Cingulate (DoR-SCC)

## Overview
This repository analyses neural recordings from six TRD patients implanted with SCCwm-DBS in order to link oscillatory power to depression severity.

The goal here is simple: recapitulate the results from my dissertation in a robust, reproducible, and elegant way.
Anything more will be pushed to a separate repository.

### Preprint
The preliminary results from this repository are [preprinted](https://www.biorxiv.org/content/10.1101/2022.07.27.501778v1.full).
Dsisclaimer: there will be updates that may or may not change the above results.

### Dissertation
This project was Aim 2/Chapter 3 of my dissertation.
See it [here](https://repository.gatech.edu/entities/publication/1f835d8c-32fd-4da9-9dfe-d4df03cc130a).

## Approach
The goal of this project was to link oscillations in bilateral subcallosal cingulate cortex (SCC) to depression severity measured with the Hamilton Depression Rating Scale (HDRS).
The resulting decoding model(s) yielded readouts, with the main one being the _depression readout from subcallosal cingulate cortex_ (DR-SCC).

### System Definitions
Defining the system is an important first step, especially in tangled problems like clinical DBS.
Here is the system diagram used for this work, presented prima fascie.

![]()


### Methods (brief)
We measured dLFP from six TRD patients implanted with SCCwm-DBS over seven months.
Oscillations calculated from bilateral SCC were then used to _decode_ depression - or find a model that correlatively-links SCC oscillatory power with depression severity 
Oscillations measured over months in the subcallosal cingulate may help us _decode_ depression - or find a measure that can help us better track the depression symptoms of patients.

This repo includes the code for Chapter 3 of my dissertation.
It is also the code for the preprint/pub []().


## Code
### Requirements
* ```dbspace``` - Available on PyPi

### Structure
* ```notebooks``` the main directory of analyses corresponding to published figures.
* ```analysis``` directory of legacy analyses. No guarantee that these scripts work
* ```.devcontainer``` contains the meta-information to run this project inside a consistent environment. Requires ```vscode``` and ```vscode::Remote Containers``` plugins; as well as ```Docker```
* ```.dvc``` directory containing meta-info for anonymized + intermediate data cloud storage
