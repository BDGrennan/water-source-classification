# **Classification Project Write-up**
## **Potable Water Classification**
#### *by William Grennan*


## Abstract
Water is one Earth's most valuable resources, and sourcing water that is drinkable is essential to every living person. Improving water supply in developing countries can massivly reduce health risks which enables a more stable environment for reducing poverty and boosting economic growth. By analyzing this dataset, we can model which water sources are potentially drinkable and which can be avoided or deprioritized. No data model can ensure water is safe to drink, so tradeoffs need to be established. How many unsafe sources can be wrongly identified as safe and how many safe sources can be ignored?


## Design
In this dataset, water sources are rated either potable as a 1 or non-potable as a 0. The models were optimized to their ROC AUC in order to provide the best opportunity for a dynamic tradeoff. In some cases users may desire a high precision, time may be limited. Other times users may need to find as many sources as possible, and testing unsuitable locations would be worth it. All models are available for viewing through the streamlit app. The app allows a user to select a model, and it will display the ROC Curve along with any hyper parameters that were used during modeling. A user is also able to see a confusion matrix, which can be modified using a number input connected to the predict probability method.


## Data
Data for the project was obtained from Kaggle as a synthetic dataset. The data includes 3,276 bodies of water with no location data attached. Each of these sources contain up to 9 measures of water quality: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, and Turbidity. Of those metrics; pH, Sulfate, and Trihalomethanes contain some empty fields and must be cleaned prior to modeling.

In the more complex forest models, all features were used about equally. So future analysis would benefit from additional simple features, such as UV sunlight exposure, type of water source (eg. stagnant pool, river, aquifer), or if macro organisms are present.

[Kaggle Water Source Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Algorithms
### Data Processing
All feature data was scaled regardless of final model to reduce human error. Data was split into 3 sets 60% was training data, 20% was validation data used to perform model selection and to tune hyper parameters, and 20% was testing data to ensure data models did not overfit.

### Data Models
Several classification models were used and scored in the creation of the streamlit app
+ Logistic Regression - 0.48 AUC
+ K Nearest Neighbor - 0.61 AUC
+ Decision Tree - 0.54 AUC
+ Extra Tree - 0.55 AUC
+ Random Forest - 0.65 AUC
+ Bagging - 0.65 AUC
+ Extra Trees - 0.64 AUC
+ XGBoost - 0.60 AUC


## Tools
Streamlit is used as an app to display information on the model used and the hyperparameters associated. It also allows a user to input a perferred prediction probability to modify the confusion matrix. Those modifications allow users to change strategy and increase or decrease the allowable false positives.

## Communications
This work along with the slides and python script has been uploaded to my Github.
