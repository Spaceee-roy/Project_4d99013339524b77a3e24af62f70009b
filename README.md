# YouTube Short Automation Script
This repository contains the code for the YouTube Short Automation Created by Spaceee-roy and helped by Attit11.

# Important Note

## CMAKE and VISUAL STUDIO CODE C++ MUST BE INSTALLED 

## PYTHON VERSION SHOULD BE 3.10 OR LOWER. CANNOT RUN
``` bash
pip install requirements.txt
```

## Needed Files

To run this project, you will need to have the following files downloaded on your device:

`Note: EXAMPLE can be any name `

`EXAMPLE.mp4`

`EXAMPLE.srt`


## Deployment

To deploy this project run

```bash
  python main.py
```


## Process

Go to the website where the code is currently running. 

Add the files required. (Check `Needed Files` above for the requirments)

The script will find a timestamp where the speaker(s) says an intresting topic.

Then it will cut that section of video and make a new .mp4 file and .mp3 file for audio.

After that, it will get the face positions of the speakers and make the video centered. If there is no face displayed, it will go to where the last face was displayed.

It will then add audio and generated subtites.

Lastly, it will burn the subtites and delete all unimportant files.

The short is now complete.

If there are multiple topics inside of the video, it will repeat these steps and generate all.
## Tech Stack

**Client:** Python

**Server:** Python



## Libraries Used

``` bash
pip install opencv-python
```
``` bash
pip install scipy
```
``` bash
pip install spacy
```
``` bash
pip install moviepy
```
``` bash
pip install datetime
```
``` bash
pip install face_recognition
```
``` bash
pip install assemblyai
```
``` bash
pip install pandas
```
``` bash
pip install sentence_transformers
```
``` bash
pip install sklearn
```
``` bash
pip install pysrt
```
