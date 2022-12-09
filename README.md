# TManual: assisting in manual measurements of length development
![time development of termite foraging tunnels](images/development_eg.png)

**TManual** is a python program that assists in manual measurements of length development (preprint: [Mizumoto 2022](https://doi.org/XXXXXXXXXX)). It can be used to measure the length of objects from sequential images, such as snapshots, timelaps, and video clips. It is designed especially for gallery structures built by animals, but can be applied to any objects. 

## How TManual works?
* Measure the length of object by just clicking on images.
* Account for the branching structure by indicating branching nodes.
* Taking over all data-handling process (scaling, zero-adjustment, mesuriment, branch structures, creating tidy dataframe), so that users can only focus on clicking without interruptions.
* Appending data for sequential images to easily trace time-development.

## How to get TManual
TManual can be installed as 1) a Python package, 2) [EXE file](standalone/tmanual_standalone.exe) for Windows users, or 3) standalone [python file](tandalone/tmanual_standalone.py).

### 1. Python package
```
pip install git+https://github.com/nobuaki-mzmt/tmanual
```
As all requirements will be installed together, I recommend using virtual environment (e.g., [Anaconda](https://www.anaconda.com/)).  
Then run the following in python.
```python
import tmanual
tmanaul.gui()
```

### 2. EXE file
Download [here](standalone/tmanual_standalone.exe).  
It is a bit heavy, but you can start TManual with just one clicking.

### 3. Standalone python file
Download [here](standalone/tmanual_standalone.py).  
As in the package, you will need to prepare requirements listed [here](requirements.txt).


## How to use TManual
There is a README on a gui interface.  
Please see detail for the preprint [(Mizumoto 2022)](https://doi.org/XXXXXXXXXX). 
![snapshot of the gui](images/gui.PNG)

## Contributor
Nobuaki Mizumoto, Okinawa Institute of Science and Technology  
Contact: nobuaki.mzmt at gmail.com

#### Acknowledgements
I thank Sang-Bin Lee for informing me that this tool is helpful for those other than myself and encouraging me to publish this tool as a paper, with valuable advice; Kaitlin Gazdick for their valuable input in designing the specifications and several sample images; and Jamie M. Kass for advice for depositing source codes. This study is supported by a JSPS Research Fellowships for Young Scientists CPD (grant number: 20J00660).
