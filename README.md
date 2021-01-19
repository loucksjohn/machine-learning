# machine-learning




! [EXO] (https://github.com/loucksjohn/machine-learning/blob/main/exoplanets.jpg?raw=true)

## DESCRIPTION

Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

To help process this data, I created two machine learning models capable of classifying candidate exoplanets from the raw dataset--which was provided by NASA.

In the notebooks within this repository, you will see how to:

1. [Preprocess the raw data](#Preprocessing)
2. [Tune the models](#Tune-Model-Parameters)
3. [Compare two or more models](#Evaluate-Model-Performance)

## Resources

In the GitHub repository for this project you will find all the resources necessary for the creation of the two machine learning models and the notebooks are there for you to run the code, as well. Here's a quick rundown of those resource files and a brief explanation:

- 'exoplanet.csv' - raw data from the NASA Kepler space program
- 'jloucks_SVC1.sav' - *Support Vector Machine(SVM)* prediction model saved to '.sav' file, using joblib
- 'jloucks_log_reg.sav' - *Logistic Regression* prediction model saved to '.sav' file, using joblib
- 'model_A_svc.ipynb' - jupyter notebook containing the code for the *Support Vector Machine(SVM)* prediction model
- 'model_B.ipynb' - jupyter notebook containing the code for the  *Logistic Regression* prediction model



## Comparison Report of the Two Models

In the end, both the SVM model and the Logistic Regression model produced roughly the same results--approximately 88% - 90% accuracy.  I began the inquiry of the exoplanet data w/ the SVM model, it was just my initial hypothesis that this model would yield the best accuracy.  Please refer to the notebook 'Model_A_svc' for analysis of the SVM model.  And with just a very basic parameter set-up, the model yielded an 84% on the training set and an 86% on the test set.  Using GridSearchCV to tune the model parameters w/ four different variables for for both the "C" & "gamma" parameters, the score was improved a bit to 88%.  So I went back and created a new model to fit, using the parameters chosen as the best by the GridSearchCV tuning.  The score for the second model (model2) yielded a score of 88% on the training and 90% on the testing.   That was a nice bit of improvement, but I wanted to try one more bit of Hyperparameter tuning to see if I could squeeze out any more. So in the second attempt at GridSearchCV, I passed in two different kernels, 'rbf' & 'linear', and tweaked both the "C" & "gamma" parameters only slightly.  However, after all of that, the score was roughly the same on this third model.....and actually it came out just a tad lower - 91% training and 89% on the testing.  So in the end, for the SVM model, I saved 'model2' as the '.sav' file since it was the one that had gotten the highest score on the testing set of data.

For the Logistic Regression model, please refer to the notebook entitled 'Model_B'.  Just like the previous model, after preprocessing the data, selecting features, removing unnecessary features, scaling the data, and separating into training and testing sets--I ran the first model w/ very basic parameters, just to see what the score would look like.  The initial model scored 85% on training and 86% on testing.  Using GridSearchCV to tune the model parameters w/ four different variables for "C" & increasing the "max_iterations" parameters, the score was improved a bit to 88%.  So I went back and created a new model ("classifier2") to fit, using the 'C' parameter chosen as the best by the GridSearchCV tuning, and also upping the 'max_iter' to 10000.  The score for the second model (classifier2) yielded a score of 89% on the training and 90% on the testing.   That was a bit of improvement, but I, again, wanted to try another round of Hyperparameter tuning to see if I could squeeze out any more. So in the second attempt at GridSearchCV, I passed in three different solver parameters, tweaked the "C" parameters based on the previous version, and included a Repeated Stratified K-Fold cross validator parameter, and a 'penalty' parameter.  With fine tuning the parameter to such a degree in the GridSearch, I really expected to see an improvement.  However, that was not the case.  Scores came back from the GridSearchCV at 88% & 89%.  So I stopped there as it seemed I was getting as good of an accuracy score as I could get and saved the 'classifier2' model as the '.sav' file.

Interestingly enough, as stated at the outset, the scores for the two models were pretty close to the same.  I honestly don't feel either model are accurate enough to predict new exoplanets.  Being up around that 90% accuracy for both models is excellent, I think, however I'm not confident at all about what's going on in the dataset.  To have full confidence in saying that either (or both) model could predict new exoplanets, I would need to understand exactly what all that data in the CSV is telling me.  That honestly the only way I would could make my models better at predicting, is to understand what is being measured in all those columns in the .csv.

One other note, I plan on going back in and doing a third model for a neural network--but I ran out of time and wanted to make sure I didn't get docked for turning it in late.  If and when I have time to update my repository with the neural network, I will also come back and add my finding to this section of the readme file.

Please contact me if you have any issues or questions.



## Author

John Loucks
Email: [johnloucks@gmail.com](mailto:johnloucks@gmail.com)
GitHub: https://github.com/loucksjohn





