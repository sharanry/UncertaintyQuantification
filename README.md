# Connection between Uncertainty Quantification and Gaussian Prior Parameters

Supplementary code for the paper on *Connection between Uncertainty Quantification and Gaussian Prior Parameters*.

### Prerequisites
 - [Python3](https://www.python.org/download/releases/3.0/)
 - [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

### Instructions

 - Clone the repository.
 ```bash
git clone https://github.com/sharanry/UncertaintyQuantification
cd UncertaintyQuantification
 ```
 - Make new `virtualenv` environment and install dependencies from `requirements.txt`
 ```bash
python3 -m virtualenv env
source ./env/bin/activate
pip install -r requirements.txt
 ```
 - Run the two scripts two reproduce results.
 ```bash
 python asymptotic_analysis.py
 python sigma_analysis.py
 ``` 
 - The generated plots will be available in `./results`.
 
### Results

#### Sigma analysis discontinuous regression using Bayesian neural networks
![](./images/sigma_ci.png)
*Fig 1(a): 95% confidence interval of the models trained over varying prior σ. Red crosses denote the data samples, blue dots represent the mean predictions and the blue fill expresses the 95% CI.*

![](./images/sigma_ess.png)
*Fig 1(b): Effective sample size of weights posterior samples at each layer in logarithmic scale.*

#### Asymptotic analysis of discontinuous regression using bayesian neural networks
![](./images/asymp_ci.png)
*Fig 2(a): 95
% confidence interval of the models trained over varying data sample sizes. Red crosses denote the data samples with prior σ = 1.0, blue dots represent the mean predictions and the blue fill expresses the 95% CI.*

![](./images/asymp_ess.png)
*Fig 2(b): Effective sample size of weights posterior samples at each layer in logarithmic scale.*

