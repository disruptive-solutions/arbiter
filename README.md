# Arbiter

Train and use a set of machine-learning models on known clean PE files and 
known malware.  Once trained, the models are used to predict the likelihood
that an unknown sample is malware.

It should be noted that this is only a library used to create and use models.
Arbiter does not come with premade models, malware, or goodware.  It is up to 
the user to find malware/goodware.  See below for some recommended 
repositories.

## Recommended Data Sets

### Malware

Finding malware is pretty easy these days.

* https://virusshare.com (requires an invite)
* https://www.virustotal.com (downloading requires a paid account)
* https://www.dasmalwerk.eu
* https://github.com/ytisf/theZoo

### Goodware

Our strategy for collecting clean samples was to use the PE files that come in
Windows as well as install some common software.  We used Ninite to create a
mega-installer and then pulled all PE files from `%ProgramFiles%` and 
`%ProgramFiles(x86)%`.  If you think of better ways, let us know.

* https://ninite.com/
* A base install of Windows OS

## Research

* https://2012.infosecsouthwest.com/files/speaker_materials/ISSW2012_Selecting_Features_to_Classify_Malware.pdf
* http://homepage.divms.uiowa.edu/~mshafiq/files/raid09-zubair.pdf
* https://link.springer.com/content/pdf/10.1186%2Fs13635-017-0055-6.pdf
* https://resources.infosecinstitute.com/machine-learning-malware-detection/
* https://github.com/carlosnasillo/Hybrid-Genetic-Algorithm

## Usage

To install Arbiter, simply run `python setup.py install` (preferrably in a 
virtual environment)

### Train

`arbiter train -m /path/to/malware/* -g /path/to/goodware/*`

Using 7 data-points from the PE structure (using [`pefile`](https://github.com/erocarrera/pefile)):
* Debug size
* Image version
* Import relative virtual address
* Export size
* Resource size
* Number of sections
* Virtual size of the second section

The train module creates the following models:
* `sklearn.linear_model.LogisticRegression`
* `sklearn.linear_model.LogisticRegressionCV`
* `sklearn.linear_model.SGDClassifier`
* `sklearn.naive_bayes.GaussianNB`
* `sklearn.svm.LinearSVC`
* `xgboost.sklearn.XGBClassifier`

Finally, `pickle` and write the models (defaults `./arbiter_models.pickle`).

### Predict

`arbiter /path/to/unknown/samples/*`

Using the models from the `train` subcommand, predict the liklihood that the 
unknown samples are malicious.  By default, this step writes a JSON file: 
`./arbiter_results.json`.  The output of the `predict` looks like this:
```
{
 "LogisticRegression": {"0": 0.6189994176326301}, 
 "LogisticRegressionCV": {"0": 0.35073582481230403}, 
 "XGBClassifier": {"0": 0.5281684994697571}, 
 "GaussianNB": {"0": 0.6666666666666666}, 
 "SGDClassifier": {"0": 0.0}, 
 "LinearSVC": {"0": 1}
}
```
The keys of the above dictionaries refer to the position of each sample in the 
input list.  This is due to change because we feel that it is not intuitive.

## Future Goals

* Group similar samples
* Ability to analyze more than just Windows PE
