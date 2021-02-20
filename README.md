# Image-and-review-based-recommender-system
This repository is created as a part of my ML2 course work.


### Steps to train a rating prediction model:
1.	Clone this repository: <br />
    ```git clone https://github.com/SayanGhoshBDA/Image-and-review-based-recommender-system.git```
2.	Enter into the directory named 'code': <br />
    ```cd Image-and-review-based-recommender-system/code/```
3.	Run the shell script named 'apriori.sh': <br />
    ```bash apriori.sh```
4.	Finally run the 'Train_and_Validate.py' to train the model: <br />
    ```python Train_and_Validate.py```

<ins>Note</ins>: Step 3 works in linux system.  For windows system, please manually run each of the commands inside the 'apriori.sh' file.  Also, if someone is using Annaconda environment, he needs to run the conda-equivalent installation commands instead of the pip-based installation commands mentioned in this shell script.  Once all the commands written in this file are executed, you will get the preprocessed data in your working directory &mdash; no need to run 'Preprocess_data.py'.

