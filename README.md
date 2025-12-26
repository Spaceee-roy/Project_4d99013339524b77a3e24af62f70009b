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
