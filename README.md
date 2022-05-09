# SeqMed
Reproducing the SeqMed work

Thanks for viewing my code that replicates the work of S.W. in SeqMed.  You'll be able to reproduce the work with the files in this directory!

1)  Get access to MIMIC-III data, and gather the following files:
+PRESCRIPTION.csv
+DIAGNOSES_ICD.csv
+PR0CEDURES_ICD.csv

2)  Download all the files in this repository to a folder of your choice.

3)  Update the directory in "data_manipulation_helpers.py" to point to the directory containing the PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv file.

4)  Open, or execute main.py.

Now you'll have saved train, test and validation data; along with 10 saved models.  You can explore and load the model for further inference!

REQUIREMENTS:
numpy
pytorch
pandas
os
sklearn
pickle
