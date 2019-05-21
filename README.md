# renamethis

Binary classifier to predict/score whether or not a binary is malware based on various attributes.

## Future Goals

* Group similar samples
* Ability to analyze more than just Windows PE

## Data Set

* https://github.com/ytisf/theZoo

## Research

* https://2012.infosecsouthwest.com/files/speaker_materials/ISSW2012_Selecting_Features_to_Classify_Malware.pdf
* http://homepage.divms.uiowa.edu/~mshafiq/files/raid09-zubair.pdf
* https://link.springer.com/content/pdf/10.1186%2Fs13635-017-0055-6.pdf
* https://resources.infosecinstitute.com/machine-learning-malware-detection/
* https://github.com/carlosnasillo/Hybrid-Genetic-Algorithm

## Tools

### Data Collector

A script to look in a directory and extract various datapoints and store them in a database

## Modeler

Take the data from the Data Collector and create a model to predict/score liklihood of an unknown sample being malware
