# SeqMed
Reproducing the SeqMed work

Thanks for viewing my code that replicates the work of S.W. in SeqMed.  You'll be able to reproduce the work with the files in this directory!

1)  Get access to MIMIC-III data, and gather the following files:
+PRESCRIPTION.csv
+DIAGNOSES_ICD.csv
+PR0CEDURES_ICD.csv

2)  Download all the files in this repository to a folder of your choice.

3)  Update the directory in "data_manipulation_helpers.py" to point to the directory containing the PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv file. (MIMICIII_FILE_PATH = '/home/...directory to PRESCRIPTIONS / PROCEDURES / DIAGNOSES files...')

4)  Open, or execute main.py.

Now you'll have saved train, test and validation data; along with 10 saved models.  You can explore and load the model for further inference!

The structure of this repository is as follows.  

-- data_manipulation_helpers.py + munge.py:  These files are sourced first to process the data and save it into pytorch dataloaders that are used in training each model.  munge.py is run first, referencing the data_manipulation_helpers.py.

-- after data processing, each model is trained in the order defined in the SeqMed work.  As each model is trained, progress reports are displayed.  Finally, each model is used to produce predictions on the held-out validation data-set.  Those predictions are saved for comparison.
    + (<model_name>.py) contains all of the code and helpers to build and train each pytorch model.  This is included for each deep-learning model.  These are unique to each model, and are quite differentiated; especially the training procedures.
    + (train_<model_name>.py) will load the helpers for each model (if required) and load the pytorch dataloaders for training, testing and validation data-splits.  It then builds and trains the model; saving the final model and testing predictions.
    
-- It's each to train just one model if you'd prefer!  open up main.py and run the "import <model_name>.py" for that model.  Remember, you'll need to have already run the "munge.py" which provides the required dataloaders.  (And include all the require modules below).

REQUIREMENTS:
numpy
pytorch
pandas
os
sklearn
pickle
