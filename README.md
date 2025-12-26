# YouTube Short Automation Script
This repository contains the code for the YouTube Short Automation Created by Spaceee-roy and supported by Attit11.

# Important Note

## CMAKE, FFMPEG and VISUAL STUDIO CODE C++ MUST BE INSTALLED 

Life will be much easier if you have administrator powers. Trust me, I found out the hard way.

## Needed Files

To run this project, you will need a directory like this:
```bash
Main Folder/
|--- temp/
|--- videos/
     |--- Video_you_want_to_process.mp4
     |--- Video_you_want_to_process2.mp4
      ...
     └--- Video_you_want_to_processN.mp4
|--- main.py
|--- executioner.py
|--- titler.py
|--- requirements.txt
|--- .env
|--- viral_clips.csv
└--- face_position.csv
```

## Deployment

To deploy this project run

```bash
  python main.py
```


## Process
Add the files required. (Check `Needed Files` above for the requirements)

Executioner.py will find a timestamp where the speaker(s) say an interesting topic.

It will find all of these interesting topics and log the top 10 in a `CSV` and generate an `HTML` for nicer visualization.

Main.py will then cut a clip, do face tracking, crop, burn subtitles and add title, description and metadata to the video.

Titler.py will be used by main.py to generate title, description and metadata.

## Tech Stack

**Python** is **goated**

## Libraries Used
To install use
```bash
pip install -r requirements.txt
```
```bash
assemblyai
tqdm
face_recognition
cv2 (OpenCV)
pandas
numpy
pysrt
groq
dotenv (python-dotenv)
spacy
sentence_transformers
keybert
transformers
bertopic
sklearn (scikit-learn)
librosa
os
sys
time
subprocess
pathlib
shutil
re
json
logging
string
tempfile
math
hashlib
datetime
typing
concurrent.futures
functools
```
