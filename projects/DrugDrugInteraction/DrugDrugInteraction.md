---
title: Drug-Drug Interaction
nav-title: Drug-Drug Interaction
category: projects
layout: default
---

The experiments can be found here: <https://github.com/linqs/psl-examples/tree/main/drug-drug-interaction>

## Drug-Drug Interaction

## Problem
In this example, we attempt to infer unknown drug drug interactions from a network of multiple drug-based similarities and known interactions.

## Dataset
There are three datasets which are derived from [DrugBank 4.3](https://go.drugbank.com/downloads/archived) and [drugs.com](Drugs.com). Each dataset below contain seven drug–drug similarities. Four of these similarity measures are drug-based: Chemical-based, Ligand-based, Side-effect-based and Annotation-based. Three similarities are between drug targets and computed by aggregating over known targets for the drugs: Sequence-based, PPI network-based, and Gene Ontology-based.
- general-interactions: This dataset has a total of 4,293 interactions across 315 drugs. There is also a file DrugBankIDs, which maps integer IDs to DrugIDs. Each fold share the same similarity data.
- crd-interactions: This dataset has a total of 10,106 CRD interactions across 807 drugs. These IDs are anonymized (so there is no DrugBankID mapping). Each fold uses different similarity data (since blocking is stricter in the crd setting).
- ncrd-interactions: This dataset has a total of 45,737 NCRD interactions across 807 drugs. These IDs are also anonymized. Each fold uses same similarity data.

## Experimental Setup
The default settings for the run script is for the dataset "general-interactions". Therefore, the evaluator thresholds must be changed when running other datasets. For example, `./run.sh -D discreteevaluator.threshold=<NEW_THRESHOLD> aucevaluator.threshold=<NEW_THRESHOLD>` will use the new values NEW\_THRESHOLD.

## Weight Learning
The original experiments this example is derived from uses an outdated weight learning method (MaxLikelihoodMPE). This example replaces the outdated weight learning method with more modern methods that perform marginally better. Using the default weights and disabling weight learning will also approximate the original results.

## Origin
This example is based on the work from [A Probabilistic Approach for Collective Similarity-Based Drug-Drug Interaction Prediction](https://linqs.org/publications/#id:sridhar-bio16). This [repo](https://bitbucket.org/linqs/psl-drug-interaction-prediction/src/master/) contains the original data and experiments from the paper. This example contains data which have been preprocessed and dumped from the original experiment. To reference the original work, please use this citation:

```
@article{sridhar:bio16,
    title = {A Probabilistic Approach for Collective Similarity-Based Drug-Drug Interaction Prediction},
    author = {Dhanya Sridhar and Shobeir Fakhraei and Lise Getoor},
    journal = {Bioinformatics},
    year = {2016},
    publisher = {Oxford},
    pages = {3175--3182},
    volume = {32},
    number = {20},
}
```

## Keywords
 - `cli`
 - `evaluation`
 - `inference`
 - `real data`
 - `weight learning`
