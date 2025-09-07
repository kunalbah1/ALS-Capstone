# ALS-Capstone

This will be a guide to understanding the GitHub Repository for the Enahcaned ALS Captsone BME Project as it was completed
during the 2024-2025 UVA Academic Year. This Capstone project ended production in May 2025.

In this Repository there should be Three Folders:

- ARDUINO CODE
- FINAL DESIGN
- PREVIOUS WORK


# Project Overview

In its simplest terms, this project aims to train a binary classifier (the .h5 file) (that is a machine learning algorithm that is
tasked with making two decisions) to determine if the video it is recieving contains an eye that in OPEN or CLOSED. This
binary classifier is trained on Black and White Open/Closed eye data and then later supplemented w/ eye images from our 
very own Capstone team.

# ARDUINO CODE:

This folder houses the ARDUINO_CODE_COMMENTED file which has the Ardunio Code we used for this project.
To upload this code into your arduino, you will need to use Arduino IDE (or somthing similar) AND install all libraries/dependencies.


After all pre-requisites are meet, the code should upload into the Arduino (via a USB) without issue.


### What Arduino Product did we use?

> We used an Arducam Mini 2MP Plus - OV2640 SPI Camera Module for Arduino UNO Mega2560 Board & Raspberry Pi Pico. There is a 
> good amount of documentation online for this model (and its setup).


# FINAL DESIGN:




This folder essentially has three files:

### CNN_noEAR_finetuned.py & CNN_noEAR.py

> Both of these files do virtually the same thing: Train the model (the file that has .h5).
> 
>So, what is the difference?
> 
> CNN_noEAR.py trains a .h5 model based off of the dataset_B_Eye_Images (Black and White Training Images)
> 
> Then CNN_noEAR_finetuned.py utilizes a process called transfer learning to retrain the existing model with images of 
> me actually wearing the BiPAP mask (found in dataset_occluded_eyes folder).
> 
> The distnction between the two .py files was done as a result of the data collection during the 
> semester being stunted by IRB denial (DO IRB EARLY!!!!). However, truthfully there is no need to
> have two separate .py files to train the .h5 model. I would reccomend to simply have one large dataset and have one .py 
> file that trains on the big dataset (that is essentially what we did here, but instead of training once on a bigger dataset
> we trained twice on two smaller ones).


### CNN_live_noEAR.py

> This file will interface with the Ardduino's live camera feed and then process the video stream
searching for VBPs (blinks that are longer than 3 seconds).
> 
> This is the file that (in theory), the end user would be benefitting from b/c it counts their VBPs.


### blink_detection_model.h5 & blink_detection_model.h5
> These files are the TRAINED MODELS (they are what CNN_live_noEAR.py will refer to in order to determine if a VBP occured).
> 
> The reason for there being two of them is the same as above:
> 
>blink_detection_model.h5 was trained on the black and white images in (dataset_B_Eye_Images)
> 
> blink_detection_model.h5 was trained using blink_detection_model.h5 AND new data from dataset_occluded_eyes. This model
> for all purposes is the model that should be feed into CNN_live_noEAR.py
> 
> Echoing a reccomendation from above, simply train once so that you only have one .h5 to work about.


This folder also has two subfolders that contain the data used to train the model.

### dataset_B_Eye_Images
> This dataset contains black and white images and was used to train a Binary Classifier (Eye == OPEN / CLOSED) to detect
> if the displayed eye was open or closed

### dataset_occluded_eyes
> This dataset contains images of me wearing the mask and was used to bolster the model trained from Black and White Images
> to 
>1. Have more data to train off of 
>2. Have a dataset that included the BiPAP mask and its potential occlusion of the eye.


# PREVIOUS WORK

If you have been wondering: "Why do the other files have 'noEAR' as a part of their name?" This folder is the reason why!

At the beginning of the semester our plan was to design the model to work on the basis of a Eye Aspect Ratio (EAR) threshold.
This idea (and the code for this folder) was heavily inspired by the work of "Adjusting eye aspect ratio for strong eye 
blink detection based on facial landmarks" (Dewi et al. 2022) [https://pmc.ncbi.nlm.nih.gov/articles/PMC9044337/].

However, we could not get a model to accurate predict the landmarks around the eyes that we wanted in time, so we pivoted to 
the Binary Classifying design that is present in the FINAL DESIGN.

This folder is a bit messy, but this was done in order to preserve as much material in the event that future developments wanted to be made.

### Synthetic_Training_Images
> There are like 34k files in this folder and as such it cannot be easily downloaded or uploaded.
> 
> The purpose of folder is to provide synthetic, randomized images of the eye region that have important landmarks (to
> be used to calculated EAR prelabelled.)
> 
> If your team wants to pursue something like this,  you can
> or generate your own (https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/tutorial.html)


### test_imgs
> These are just a subset of Synthetic Training Images made for ease of testing


### CNN2_live_detection.py
> This file is essentially the same as CNN_live_noEAR.py except instead of predicting based off [Open/Closed],
> it predicts based off whether the EAR passes a certain threshold (it is more sensitive, but could provide more nuance to the design).
> 
> DOES NOT WORK


### CNN_eye_landmarks.py
> Once again is analogous to CNN_noEAR_finetuned.py & CNN_noEAR.py except it trains a .pth model based off of landmark detection.
> 
> DOES NOT FULLY WORK


### shape_predictor_68_face_landmarks.dat
> This file was NOT made by this capstone team. It comes from the work of Dewi et al. (2022) and contains a model for
> the detection of a FULL FACE



# IMPROVEMENTS
> * IRB SUBMISSION: This cannot be stressed enough, do not slack on IRB. During my semester my team (and a lot of my friends teams)
> were unable to do major parts of our projects because we did not get the IRB done in time. UVA has IRB Office Hours
> and we have provided our IRB materials to you. PLEASE be on top of and proactive about IRB.
> -------------
> * HARDWARE: The camera that we used for our project had a servicable fram rate (~10 frames per second); however, this 
> number should be improved to see a signficantly improved product. The camera we used also had visual issues occasionally
> were the screen would freeze or become polluted by colors (like TV static). Our estimation was due to cheap hardware 
> and unstable connection. Get a better camera!!
> ------------
> * SOFTWARE: The software we have provided here should be a good building off point. Whether you decide to continue 
> with the binary classifier or go forth with the EAR Threshold (both have merit in my opinion) is up to you.
> However, improvements should be made to the training process + dataset and bugs within the hardware interface / general
> bugs worked out.


Everything here is, of course, a suggestion. I hope that there is something in here that is benefical to your
capstone group. Good Luck!