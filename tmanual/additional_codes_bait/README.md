# Additional code for bait analysis in TManual

This is an additional code for the paper,  
**Subterranean termites (Coptotermes formosanus [Blattodea: Rhinotermitidae]) colonies can readily intercept commercial inground bait stations placed at label-prescribed distance**  
by **Kaitlin Gazdick, Sang-Bin Lee, Nobuaki Mizumoto, Thomas Chouvenc, and Nan-Yao Su**  
  
This study examined how termites discover bait stations that are used to manage structural pest subterranean species.  
The study recorded the tunnel developments of colonies of *Coptotermes formosanus* in a large arena (3.68 m × 1.2 m) and measured the tunnel developments using TManual.  
This additional code overlaid hypothetical baits between real baits to study how increased baits affect the bait discovery time.  
See the paper for more details.

## How to run
First, open the virtual environment with the installed package TManual.  
Run the program 
```
python path-to-bait_analysis.py
```
It will open a GUI to determine the bait size and positions. 
![snapshot of the gui](tmanual/additional_codes_baits/GUI.PNG)
* The input folder should be the same as that used for TManual analysis.  
* It will open the first image and ask you to click the initial starting point of tunnel development and select baits.
* Then run the analysis automatically. All results will be saved in the output folder (same with that for TManual analysis in default)


## Contributor
Nobuaki Mizumoto, Auburn University
Contact: nzm0095@auburn.edu
