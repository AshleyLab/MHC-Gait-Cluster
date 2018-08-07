# Gait Analysis

This summer project centers around using 6MWT triaxial accelerometer data on doing person authentication, sex prediction, and BMI class prediction. The models present at the moment are KNN and Random Forest. Future expectations incude improving performance of these tasks as well as using gait to predict Cardiovascular Disease (CVD).

## Getting Started

There are two main Jupyter notebooks in this project. MHC-Gait-Data is the notebook for data cleaning and analysis. MHC-Gait-Models is for all ML models. The folder structure for the project is as follows:

```
/data/
```
This is where all demograhpic, CVD data, and files for testing are held.

```
/python/
```
This is where all scripts for Sherlock Jobs are located. Computationally intensive scripts should be in here.

```
/plots/
```
This is where all plots and saved figures should output to. 

In MHC-Gait-Data v4 corresponds to Synapse table (v4-v2), v2 corresponds to Synapse table (v2), v4-v1 corresponds to Synapse table (v1), and v6 corresponds to Synapse table (v6-v4).

### Prerequisites

All libraries used in notebooks must be installed for the user on Sherlock. Change seaborn to applicable package.

```
pip3 install --user seaborn
```

## Contributing

Maintain the organization of the project.

## Authors

* **Bhargav Yadavalli** - *Initial work* - [Bhargav Yadavalli](https://github.com/bhargav-y)

## Credit

This project was done at Dr. Euan Ashley's Lab at Stanford University.

## Acknowledgments

* Jessica Torres Soto
* Steve Hershman
* Anna Shcherbina
