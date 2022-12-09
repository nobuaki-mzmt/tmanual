# TManual: assisting in manual measurements of length development
![time development of termite foraging tunnels](images/development_eg.png)

**TManual** is a python program that assists in manual measurements of length development (now available as preprint [(Mizumoto 2022)](https://doi.org/XXXXXXXXXX)). It can be used to measure the length of objects from sequential images, such as snapshots, timelaps, and video clips. It is designed especially for gallery structures built by animals, but can be applied to any objects. 

## Get TManual
TManual can be installed as 1) a Python package, 2) [EXE file](standalone/tmanual_standalone.exe) for Windows users, or 3) standalone [python file](tandalone/tmanual_standalone.py).

### Python package
```
pip install git+https://github.com/nobuaki-mzmt/tmanual
```
As all requirements will be installed together, I recommend using virtual environment (e.g., [Anaconda](https://www.anaconda.com/)).
Then run the following in python.
```python
import tmanual
tmanaul.gui()
```

### EXE file
click here to download
<a id="raw-url" href="https://raw.githubusercontent.com/nobuaki-mzmt/tmanual_standalone.exe">Download FILE</a>



### Standalone python file
click here to download



## Who wants to use TManual?
Do you want to measure the length of something from many images? (especially sequential images for time develoments?) Here is TManual for you. TManual provides a user interface to click to indicate the shapes of the objects and take over all other processes, including scaling the unit, zero-adjustment, mesuring the length, and assigning nested structures (if you work of gallery systems).

## How TManual works?
TManual displays all the images in input foldier, and what users do is just clicking points of interest. TManual automatically creates a CSV file storing all information about length of structures.

## Table of Contents of repository
* [README](./README.md)
* [tmanual](./tmanual) - modules for TManual
  * [gui.py](./tmanual/gui.py)
  * [image.py](./tmanual/image.py) - module with functions and class
  * [main.py](./tmanual/main.py)
  * [measurement.py](./tmanual/measurement.py)
  * [postanalysis.py](./tmanual/postanalysis.py)
* [test](./test) 
  * [images]

## Contributor
-Nobuaki Mizumoto, Okinawa Institute of Science and Technology

## Acknowledgements
I thank Sang-Bin Lee for informing me that this tool is helpful for those other than myself and encouraging me to publish this tool as a paper, with valuable advice; Kaitlin Gazdick for their valuable input in designing the specifications and several sample images; and Jamie M. Kass for advice for depositing source codes. This study is supported by a JSPS Research Fellowships for Young Scientists CPD (grant number: 20J00660).