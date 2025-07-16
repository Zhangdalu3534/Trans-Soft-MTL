# Trans-Soft-MTL
Crisis prediction of online public opinion on representative natural disaster events in micro-blog platform
## Overview
Trans-Soft-MTL is a deep learning model for predicting the crisis degree of online public opinion of multiple natural disasters.

## Requirements 
-Text processing
jieba>=0.42.1
emoji>=2.2.0
textblob>=0.17.1
regex>=2021.4.4

-Data manipulation and analysis
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.2
scipy>=1.7.3
statsmodels>=0.13.2

-Data visualization
matplotlib>=3.5.1
seaborn>=0.11.2
plotly>=5.5.0

-Graph/network analysis
networkx>=2.6.3
python-igraph>=0.10.4
leidenalg>=0.8.10

-Machine learning and modeling
xgboost>=1.6.2
catboost>=1.0.6
shap>=0.41.0
bayesian-optimization>=1.4.3

-Deep learning
torch>=1.10.0

## Installation
1.Clone the repository:
https://github.com/Zhangdalu3534/Trans-Soft-MTL
2.Install the required packages:
pip install -r requirements.txt

## Usage
The data used in this study is in a folder called“rawdata”

## To run the model:
1.Prepare your ND-OPO dataset in the required format.
2.Train the model using the provided script.
For more details on usage, check the documentation in the codebase.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
