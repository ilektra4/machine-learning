This repository contains the Python scripts used to train and evaluate the best deception detection model using XGBoost. 

The main files for the final results are:

- 0_Youtube_Extractions
- Final_Report.ipynb

All other folders (1_Preprocessing, 2_Models, aggregation_and_models, and test_logistic_stacking) contain additional experiments and intermediate tests we ran.

The model was trained on the Kaggle dataset with the selected hyperparameters: 
- 600 estimators
- maximum depth of 3
- learning rate of 0.05
- with 5-fold stratified cross-validation.
  
The trained model was used to generate predictions on a separate test dataset (dolos dataset). 
On the test dataset, the model achieved an accuracy of 0.651 and an AUC of 0.682. 
On the Kaggle dataset, the model achieved an accuracy of 0.583 and an AUC of 0.667. 

All training and prediction steps are implemented in the provided XGBoost_Training_and_Generalisation_FINAL.ipynb file, which were used to obtain these results.
