# Connection between Uncertainty Quantification and Gaussian Prior Parameters

Supplementary code for the paper on *Connection between Uncertainty Quantification and Gaussian Prior Parameters* submitted to *Bayesian Deep Learning workshop at NeurIPS'19*.

### Prerequisites
 - [Python3](https://www.python.org/download/releases/3.0/)
 - [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

### Instructions:

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
 - Run the two two scripts two reproduce results.
 ```bash
 python asymptotic_analysis.py
 python sigma_analysis.py
 ``` 
 - The plots generated will be available in `./results`.
 
