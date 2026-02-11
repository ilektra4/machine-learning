This repository contains the Python scripts used to train and evaluate the best deception detection model using XGBoost. 
The main files for the final results are:
-0_Youtube_Extractions
-the main script/notebook (main file)
All other folders (1_Preprocessing, 2_Models, aggregation_and_models, and test_logistic_stacking) contain additional experiments and intermediate tests we ran.

The model was trained on the Kaggle dataset **name_to_be_added** with the selected hyperparameters: 
- 600 estimators
- maximum depth of 3
- learning rate of 0.005
- with 5-fold stratified cross-validation.
The trained model (**name_to_be_added**) was used to generate predictions on a separate test dataset. On this dataset, the model achieved an accuracy of **to_be_added***% and an AUC of 0.690. 
The confusion matrix and classification report indicate that while the model performs slightly better than random guessing, it shows limited generalization on the new data.

All training and prediction steps are implemented in the provided **name_to_be_added**.py files, which were used to obtain these results.
