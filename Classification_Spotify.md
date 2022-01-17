---
title: Classification & Billboard 100


---



## **Predicting Hit Songs using Machine Learning Classification**

In my supervised machine learning project as part of the Metis Data Science Bootcamp I built a classification model to predict whether a song will be a hit based on its audio features.



**Introduction**

A hit song can make an artist’s career, and make millions for record labels. While hit songs have certain tropes, such as upbeat and catchy, not all songs that fit this bill go viral. 

The goal of this project is to use machine learning classification to predict whether a song will be a hit depending on its audio characteristics. With a data-driven approach to evaluating a song, record labels and artists will be better equipped in their music production moving forward.



**Design**

To predict whether a song will be a hit, numerous audio features will be explored. The correlation between the features and whether a song is a hit, as well as the correlation between each other for potential feature engineering, should be considered.



After data cleaning, feature engineering, and data manipulation, I split the data into training and test sets, and tested five different models: logistic regression, decision trees, random forests, ADA Boost, and XG Boost. I modeled based on the training set and judged the efficacy of the models on the test data. I also tuned the hyperparameters of the models via `GridSearchCV` and tuned the probability threshold via `predict_proba` in Sklearn.



The main metrics I used to judge the models are F1 score and ROC-AUC. Given the costs associated with producing and marketing music, it is very important that there are few false positives. That said, it is also important to not miss out on a golden opportunity, making precision important as well. For this reason, I will use F1, as this incorporates both precision and recall for a more holistic view of model performance. In addition, ROC-AUC does a great job of measuring how effective a model is in finding hits while simultaneously avoiding false positives.



**Data**

This [Kaggle Dataset](https://www.kaggle.com/multispiros/34740-hit-and-nonhit-songs-spotify-features) contains data on over 34,000 songs extracted from Spotify, dating back to 1985. Luckily, the dataset used was already curated and cleaned. Out of the 34,000+ data points, half were non-hits, and the other half were hit songs. (A song qualifies as a hit if it ever appeared on the Billboard 100 or Spotify 100 list.)

Further information on the fields can be found on the  [Spotify for Developers](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features) website.



**Algorithm**



![corr plot](https://github.com/prathapr91/classification_spotify/blob/main/Images/Screen%20Shot%202022-01-13%20at%202.04.02%20PM.png?raw=true)

![pairplot](https://github.com/prathapr91/classification_spotify/blob/main/Images/pairplot.png?raw=true)

Based on the above correlation matrix and pairplot, I was able to discern patterns and trends. Using these, I simplified the model by taking in the ones most relevant to its hit status as well as feature-engineering new variables that account for trends observed. Right off the bat, I was very surprised to see that the “danceability” variable had minimal effect on whether or not a song is a hit.

*Variable Selection*

- After analyzing the correlation matrix and pairplots of the dataset and its interaction with the target variable, I narrowed down the variables to 7 baseline variables and 3 feature-engineered variables. The baseline variables are as follows:
  - Duration of the track in minutes
  - Energy, ranging from 0 to 1.0, which measures intensity and activity
  - Acousticness, ranging from 0 to 1.0, which measures how acoustic the track is
  - Instrumentalness, ranging from 0 to 1.0, where values closer to 0 are more vocal and values closer to 1.0 are more instrumental. For example, rap music will typically be on the lower end of the scale while classical music will typically be on the higher end of the scale.
  - Valence, ranging from 0 to 1.0, which detects the happy and cheerful emotions in a track
  - Loudness, which detects the overall loudness of a track in decibels
  - Speechiness, ranging from 0 to 1.0, which detects the presence of spoken words in a track. Values closer to 1.0 are more spoken than music (I.e. a talk show), whereas values closer to 0 contain little speech and are mostly music.

*Feature Engineering*

Variables Added:

- time_signature dummy variable. 4 beats per measure is a strong indicator of whether a song is a hit, as 16,588 out of 17,425 hit songs in the dataset are 4 beats per measure.
- lowloud indicator. There is a steep drop off in positives identified below -10dB, so adding an indicator variable here would boost the algorithm.
- EnergyxDance interaction term. Hit songs are predominantly pop or hip hop, which are genres with songs that have high danceability and energy.



**Results/Analysis**

GridSearchCV was used on each model to tune hyper parameters and threshold tuning was used to optimize F1. After this was done, the train and test F1 scores were compared for quality and fit, and an ROC-AUC comparison was used to get a holistic sense of model performance. Results:

| Model         | Train F1 | Test F1 |
| ------------- | -------- | ------- |
| Logistic      | 0.792    | 0.788   |
| Decision Tree | 0.868    | 0.777   |
| Random Forest | 0.919    | 0.826   |
| ADA Boost     | 0.805    | 0.803   |
| XG Boost      | 0.845    | 0.824   |

![ROC_Curve](https://github.com/prathapr91/classification_spotify/blob/main/Images/ROC%20curve.png?raw=true)



The two best performing models are the Random Forest and XGBoost. Ultimately, I went with XGBoost due to mitigated overfitting concerns. At a probability threshold of 0.381, the F1 score is 82.4% and the ROC-AUC is 0.891. The F1 score increases as the probability threshold increases but beyond the .381 threshold it tapers slightly. Below is a chart outlining the performance of XGBoost at all thresholds, up to the default threshold of .5:

![Prob_threshold_XG](https://github.com/prathapr91/classification_spotify/blob/main/Images/Prob_Threshold_XG.png?raw=true)

Please see below for a confusion matrix for further analysis:

![Confusion Matrix_XGBoost](https://github.com/prathapr91/classification_spotify/blob/main/Images/Confusion%20Matrix_XGBoost.png?raw=true)



**Model in Action**

As an example of this model performing accurately (and not accurately), I sampled a few songs and compared it against what my algorithm predicted.

![Examples](https://github.com/prathapr91/classification_spotify/blob/main/Images/Screen%20Shot%202022-01-13%20at%203.16.29%20PM.png?raw=true)



This model did a great job in correctly predicting hits and non-hits, as well as avoiding false negatives. For example, the song Goosebumps by Travis Scott incorporates all the traditional audio components of a hit and is correctly identified as so. One limitation of this model is that it does not contain artist social media popularity or reputation data. For example, several songs from Drake, one of the most popular artists of this generation, have low energy and valence but are considered hits in large part due to his reputation. 

**Conclusion**

The XGBoost model did a great job in correctly predicting hits and non-hits, as well as avoiding false negatives. However, there were quite a few false negatives determined from this model. One limitation of this model is that it does not contain social media popularity. Incorporating social media following into the model would be an excellent addition as this variable has a high correlation with success.



To see my project in further detail, please visit my [GitHub Repo](https://github.com/prathapr91/classification_spotify).



**Tools**

- Pandas for data manipulation
- Numpy and Scikit-learn for statistical analysis
- Matplotlib and Seaborn for visualization

