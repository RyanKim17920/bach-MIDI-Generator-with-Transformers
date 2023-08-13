# MIDI Generator with Transformers
##### Using Machine Learning and Transformers in Python to generate classical music MIDI files.

This is currently a work in progress (WIP) of the music generator. 
As of now, I am still trying to create a MIDI to numpy array convertor.

As this project is currently unfinished, the repository will be a highly changing and active.
If you have any solutions to issues that occur within the code, please request a merge request.
Thanks! 

The MIDI files were extracted from: http://www.jsbach.net/midi/ (Yes, I got permission from the owner of this website, Dave, to modify the MIDI files for my needs).

#### Current workings: Transformer model and final Data Preparation!

#### To do:
- [x] Create a MIDI to numpy array convertor
- [x] Create a numpy array to MIDI convertor
- [ ] Extract all Data
- [ ] Create a Transformer model
- [ ] Create generalization of extraction
- [ ] Create a GUI for the model

#### Note:
If you are using your own MIDI files, and they are from the internet, they may be corrupted.
Luckily, there is a way to fix this by uploading to MuseScore Desktop and re-exporting back. 
Somehow, this fixes errors in the MIDI files.