# PetFinder.my - Pawpularity Contest

## Problem

Currently, PetFinder.my uses a basic [Cuteness Meter](https://www.petfinder.my/cutenessmeter) to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved.

## Goal

Solution that improves the appeal of pet profiles, automatically enhancing photo quality and recommending composition improvements. As a result, stray dogs and cats can find their "furever" homes much faster.
Our end goal is to deploy AI solutions that can generate intelligent recommendations (i.e. show a closer frontal pet face, add accessories, increase subject focus, etc) and automatic enhancements (i.e. brightness, contrast) on the photos, so we are hoping to have predictions that are more easily interpretable.

## Evaluation

Mertic RMSE, the lower the better.

For each Id in the test set, you must predict a probability for the target variable, Pawpularity. The file should contain a header and have the following format:

```txt
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Submission

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named submission.csv

## Data description
Raw images and metadata to predict the “Pawpularity” of pet photos.

#### Purpose of Photo Metadata

-  **Photo Metadata**, data related to each photo for key visual quality and composition parameters.
- These labels are  **not used**  for deriving our Pawpularity score, but it may be beneficial for better understanding the content and co-relating them to a photo's attractiveness. 
- In PetFinder production system, new photos that are dynamically scored will not contain any photo labels. If the Pawpularity prediction model requires photo label scores, PetFinder will use an intermediary model to derive such parameters, before feeding them to the final model.

### Training Data

- **train**  - Folder containing training set photos of the form  **{id}.jpg**, where  **{id}**  is a unique Pet Profile ID.
- **train.csv**  - Metadata (described below) for each photo in the training set as well as the target, the photo's  **Pawpularity**  score. The  **Id**  column gives the photo's unique Pet Profile ID corresponding the photo's file name.

#### Example Test Data

In addition to the training data, some randomly generated example test data included to help author submission code.
- **test/**  \- Folder containing randomly generated images in a format similar to the training set photos. The actual test data comprises about 6800 pet photos similar to the training set photos.
- **test.csv**  \- Randomly generated metadata similar to the training set metadata.
- **sample_submission.csv**  \- A sample submission file in the correct format.

#### Photo Metadata

The  **train.csv**  and  **test.csv**  files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of  **1**  (Yes) or  **0**  (No) for each of the following features:

- **Focus**  \- Pet stands out against uncluttered background, not too close / far.
- **Eyes**  \- Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face**  \- Decently clear face, facing front or near-front.
- **Near**  \- Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action**  \- Pet in the middle of an action (e.g., jumping).
- **Accessory**  \- Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group**  \- More than 1 pet in the photo.
- **Collage**  \- Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human**  \- Human in the photo.
- **Occlusion**  \- Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info**  \- Custom-added text or labels (i.e. pet name, description).
- **Blur**  \- Noticeably out of focus or noisy, especially for the pet’s eyes and face. For Blur entries, “Eyes” column is always set to 0.

## Presentation
https://docs.google.com/presentation/d/1XQAzKY56h-ATp-IHJ9xtEhmOnpXwEoIeAxSe-SD2zXQ/edit?usp=sharing

