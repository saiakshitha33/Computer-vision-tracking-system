# Project :

Hi,

check out 
- `result_final.mp4`. 
- `result_bytetrack.avi`
- `result_strongsort.avi`


## Problem statement :

Build an AI-based system that can re-identify persons in a video sequence/or live webcam that have been temporarily lost due to occlusion or change in appearance. The system should be able to track the persons even when they are partially or completely occluded, and re-identify them when they reappear. The person should keep the same ID even if they leave the frame and come back again. Count the total number of  persons seen in the image.

----

This submission is very quick and without deep diving into anything. Tracking the objects is no major challenge with the open source content we have. As you mentioned the detector, or tracker is not the important thing of the challenge but  when "person leave the frame and come back again" and gets assigned to the same ID is. I wanted to work on that specific problem statement. 




- Some months ago, i worked on similar problem statement but i had to deal with using simplee low computing algorithms. There i detected a person and saved its photo in some database. and When that person left the frame and new person got detected. I basically took the euclidean distance of the detected person and the existing person. If it's above some calibrated threshold. It'll get assigned as a new ID and so.

### Run Your Own :

If you open the file, you'll see 
```
TRACKER  =  "bytetrack"
```
switch to `strongsort`. 

i should correct myself. bytetracker didn't need reid weights. strongsort did. i've included that as well in the code.

```
python yolov8_util.py
```

as you said, applying trackers on the top of your code. I tried both bytetracker and strongsort as you can see them in *.avi files stored in the same directory.

It was holi festival here in india. Sorry for the delay.

Happy Holi! :)



### References :

https://github.com/mikel-brostrom/yolov8_tracking
