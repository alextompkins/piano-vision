# Piano Vision #
Automatic transcription and assisted tutoring for amateur piano players. 

## How to Run
1. Run `pip install .` to install OpenCV and numpy depedencies.
2. Use `python run.py` to run the program. The only parameter is which video (from `./data`) that you want to use, i.e. execute `python run.py call_me_maybe` to use that video instead. 
3. Use `python calc_accuracy.py` to calculate accuracy statistics for the output after running. This will parse the generated logs found in `./output`.
4. Several displays will be shown, but the main one to watch is named `keyboard`.

## Program Structure
* The main class of the program is found in `piano_vision/main.py`. 
* But most processing is done in the classes found in `piano_vision/processors`. Essentially, each stage of the pipeline has its own class. 
* Test videos can be found in `./data`.
* Ground truths for these videos can be found in `./ground_truths`.

## Install Instructions
### Production ###
To do a production install, run `pip install .` or `pip install -r requirements.txt`.

### Development ###
To install the package for development, run `pip install -e .[dev]`. This installs all the regular required packages and those listed in the dev section of extras_require.

After adding any dependencies to `setup.py`, regenerate the `requirements.txt` file with `pip-compile`.