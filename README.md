# TManual: Assistance for manually measuring length development of animal structures
![time development of termite foraging tunnels](images/output.gif)

**TManual** is a Python program that assists in manual measurements of length development ([Mizumoto 2023, Ecol Evol](https://doi.org/10.1002/ece3.10394)). It can be used to measure the length of objects and extract network structures from sequential images, such as snapshots, time-lapses, and video clips. It is designed especially for gallery structures built by animals but can be applied to any other objects. 

## How TManual works?
* Measure the length of the object by just clicking on images.
* Inferring the branching structure and reconstructing the network.
* Taking over all data-handling processes (scaling, zero-adjustment, measurement, branch structures, creating tidy dataframe), so that users can only focus on clicking without interruptions.
* Appending data for sequential images to easily trace time-development.
* **Realize stress-free and efficient manual measurement of a large number of images.**

## How to get TManual
TManual can be installed as 1) a Python package, 2) an [EXE file](standalone/tmanual_standalone.exe) for Windows users, or 3) a standalone [python file](standalone/tmanual_standalone.py).

### 1. Python package
This project has been tested with Python 3.9. 
```
pip install git+https://github.com/nobuaki-mzmt/tmanual
```
As all requirements will be installed together, I recommend using a virtual environment (e.g., [Anaconda](https://www.anaconda.com/)).  
Here is an example of creating a virtual environment for TManual.
```
conda create --name tmanual python=3.9
```
If this does not work, you may want to update your pip and setuptools:
```
python -m pip install --upgrade pip
pip install --upgrade setuptools packaging
```

Then run the following in Python.
```python
import tmanual
tmanual.gui()
```

### 2. EXE file
Download [here](standalone/tmanual_standalone.exe).  
It is a bit heavy, but you can start TManual with just one click.

### 3. Standalone python file
Download [here](standalone/tmanual_standalone.py).  
As in the package, you will need to prepare the requirements listed [here](requirements.txt).

## How to use TManual
Please see the detail for the preprint [(Mizumoto 2022)](https://doi.org/10.1101/2022.12.21.521503). 
* **Measurement**  
Show all images sequentially and gets user input to create res.picke. Measurement consists of the following process:  
1. Check  
Users decide the action for the desplayed image
2. Ref point  
Indicate the reference point. The reference point is an identifiable landmark across all images (e.g., the corner of the experimental arena). This is useful when the relative position of the camera and object is not fixed (e.g., when users take photos every 24 hours and need to bring the experimental arena under the camera when filming). If the camera and object are fixed, users can skip the process (The Ref point will be a left-above corner of the image).
3. Measure  
Users draw the galleries as freeform line objects with straight segments. Branching galleries should be contact with previous galleries.
4. Scale  
Measure the length of the scale object. This is used to convert the unit from pixel to mm during the post-analysis.

* **Post-analysis**  
Create CVS files containing all of the information about the gallery structures based on res.pickle.  
This includes the length of each gallery (and total length), the number of galleries, the number of nodes, gallery classification, and network structure of gallery system.

There is a Manual on a GUI interface.  
![snapshot of the gui](images/gui.PNG)

## Citation
Please cite this work as:  
Mizumoto N. 2023. "TManual: Assistant for manually measuring length development in structures built by animals". Ecology and Evolution 13(8):e10394. https://doi.org/10.1002/ece3.10394

## Contributor
Nobuaki Mizumoto
Contact: nobuaki.mzmt at gmail.com
