# Bayes
Bayes is a dynamic Bayesian network model for analysing interspecies relationships of microorganisms using python.<br>  Functions for constructing dynamic Bayesian network models using Bayes: [https://github.com/Wangy157/Bayes/tree/main/Bayes]( https://github.com/Wangy157/Bayes/tree/main/Bayes)<br>  Examples on using pgmpy: [https://github.com/Wangy157/Bayes/tree/main/Example]( https://github.com/Wangy157/Bayes/tree/main/Example)<br>  Data used in this model：[https://github.com/Wangy157/Bayes/tree/main/Data]( https://github.com/Wangy157/Bayes/tree/main/Data)<br>  Validation of the Bayes model：[https://github.com/Wangy157/Bayes/tree/main/Validation]( https://github.com/Wangy157/Bayes/tree/main/Validation)

## Dependencies  pgmpy has the following non-optional dependencies:<br>  

python 3.6 or higher<br>  network<br>  scipy<br>  numpy<br>  pytorch<br>  pgmpy<br>  sklearn<br>  pandas<br>  matplotlib<br>  


## Installation
To install Bayes from the source code:  Open the git command line and type:
```
git clone https://github.com/Wangy157/Bayes
```

To install Bayes from zip.  Click *Download Zip* to download the zip file and unzip it  

## Usage  

The Microbialprediction function is defined in the Bayes package and can be used to learn dynamic Bayesian networks directly when using Bayes.  

```
from Bayes import MicrobialPrediction
MicrobialPrediction.getBayes("###", 0.2)
```  

where #### is the name of the original data file and 0.2 is the sampling rate of the original data file

## Examples  

Example of data filtering using THSD_code in Example for a set of 12626 prokaryotic OTUs and 1595 eukaryotic OTUs, and learning a dynamic Bayesian network on the filtered data

## Citing  

```@inproceedings{ankan2015pgmpy,
  title={pgmpy: Probabilistic graphical models using python},
  author={Ankan, Ankur and Panda, Abinash},
  booktitle={Proceedings of the 14th Python in Science Conference (SCIPY 2015)},
  year={2015},
  organization={Citeseer}
}
```
