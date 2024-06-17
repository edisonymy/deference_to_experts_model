# TO DO

# classes

**Ideal_Credences**
The Ideal_Credences class is designed to map propositions to credences, representing an ideal epistemic state where the truth of a proposition is binary, but the credence (degree of belief) ranges from 0 to 1. This class assumes that credences are sharp and unique, reflecting the most rational belief an agent could have given all available evidence. It includes methods to initialize the class with propositions and to randomly generate ideal credences for those propositions.

**Expert_Group**
The Expert_Group class models a group of experts whose beliefs are represented at the group level. It simulates the individual judgments of experts based on an ideal credence function, accounting for biases and variances within the group. This class does not update the expert group's credences but provides aggregate judgments based on the individual expert opinions. Methods include simulating individual judgments, calculating aggregate judgments, and plotting these judgments for analysis.

**Bayesian_Agent**
The Bayesian_Agent class represents an agent using Bayesian inference to estimate ideal credences. The agent uses expert group judgments to update its beliefs about the ideal credences. This class includes methods for setting up expert models with different prior distributions, generating these models, and running MCMC sampling to obtain posterior distributions of the ideal credences. It also provides visualization tools for analyzing the inference process