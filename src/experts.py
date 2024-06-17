import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import name_generator
import scipy.stats as stats
from typing import List
from plotting import plot_all_individual_judgements
from ideal_credences import Ideal_Credences
class Expert_Group():
    '''
    Experts are represented at the group level, questions of belief merging and aggregation 
    within the group are abstracted. individual experts are simulated for fun.

    Expert groups are static in this model. Their credences do not update

    Expert groups are defined relative to an ideal credence function (since the attribute "bias" is relative to ideal credence)
    '''
    # i don't need population_simulation because np.sample(n_experts) will already get me central theorem theorem.
    def __init__(self, ideal_credence_obj:Ideal_Credences, name = None, n_experts:int = 100, population_sd:float = 0.2, population_bias:float = 0.1, simulate = True, plot = False) -> None:
        ''' Initializes the Expert_Group class with given parameters and optional simulation and plotting. '''
        if name is not None:
            self.name = name
        else:
            self.name = name_generator()
        # simulation settings
        self.n_experts = n_experts
        # each expert population is defined by two attributes: bias, which is relative to an ideal credence, and population_sd, 
        #which is the standard deviation of the opinions among thepopulation
        self.population_bias = population_bias
        self.population_sd = population_sd
        # get standard deviation of sample mean using central limit theorem
        self.sample_mean_sd = self.population_sd / np.sqrt(self.n_experts)
        # get sample standard deviation

        self.ideal_credences = ideal_credence_obj.credence_function
        self.propositions = list(self.ideal_credences.keys())
        # each expert group has a set of aggregate judgements, which are functions that map credences to propositions
        self.aggregate_judgements = {} 
        self.sample_sd_dict = {}
        self.individual_judgements = {}
        # run init methods
        if simulate:
            self.simulate_expert_individual_judgements()
            self.get_aggregate_judgements()
        if plot: 
            self.plot_individual_judgements()
            self.plot_expert_sample_mean_distribution()
    
    def simulate_expert_individual_judgements(self, plot = False):
        ''' Simulates individual expert judgments based on the ideal credences, accounting for bias and standard deviation. '''
        # randomly select n_experts_sample experts from self.expert_population_judgements, which is a dictionary of arrays
        keys = self.propositions
        values = np.array(list(self.ideal_credences.values()))
        # randomly generate a single sample mean using the sample_mean_sd centered around bias + ideal credence
        self.population_means = {prop:value for prop, value in zip(keys, self.population_bias + values)}
        self.individual_judgements = {prop: np.clip(np.random.normal(loc=val, scale=self.population_sd, size=self.n_experts), 0, 1)
                    for prop, val in zip(keys, self.population_means.values())}
        self.sample_sd_dict = {key: np.std(val) for key, val in self.individual_judgements.items()}
        if plot:
            self.plot_individual_judgements()

    def get_aggregate_judgements(self):
        ''' Calculates the aggregate judgments for each proposition by averaging the individual expert judgments. '''
        # for each proposition, get the mean of the sampled expert judgements
        self.aggregate_judgements = {key: np.mean(val) for key, val in self.individual_judgements.items()}

    def plot_individual_judgements(self):
        ''' Plots the individual expert judgments for each proposition as histograms. '''
        
        n_props = len(self.propositions)   
        plt.figure(figsize=(12, 3 * n_props))

        for i, (prop, values) in enumerate(self.individual_judgements.items(), start=1):
            label = f'{self.name} (Bias: {self.population_bias}, population sd: {self.population_sd:.3f}, sample mean sd: {self.sample_mean_sd:.3f}, sample_sd = {self.sample_sd_dict[prop]:.3f})'
            plt.subplot(n_props, 1, i)
            sns.histplot(values, kde=False, bins='auto', color='skyblue', label=label, stat="density")
            plt.axvline(x=self.aggregate_judgements[prop], color='b', linestyle='--', label=f'Aggregate judgement')
            plt.axvline(x=self.ideal_credences[prop], color='black', linestyle='--', label=f'Ideal credence')
            plt.legend()
            plt.title(f'Samples Distribution for {self.name} judgements on "{prop}"')
            plt.xlabel('Value')
            plt.xlim(0,1)
            plt.ylim(0,15)
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_expert_sample_mean_distribution(self):
        ''' Plots the distribution of expert sample means for each proposition. '''
        n_props = len(self.propositions)
        plt.figure(figsize=(12, 3 * n_props))
        for i, prop in enumerate(self.propositions, start=1):
            plt.subplot(n_props, 1, i)
            label = f'{self.name} (Bias: {self.population_bias}, sample mean sd: {self.sample_mean_sd:.3f})'
            mu = self.population_means[prop]
            sigma = self.sample_mean_sd
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma),label=label)
            plt.axvline(x=self.ideal_credences[prop], color='black', linestyle='--', label=f'Ideal credence')
            plt.xlim(0,1)
            # plt.ylim(0,15)
            plt.legend()
            plt.title(f'Expert sample mean distribution for "{prop}"')


#### Questions
        '''
        1. should expert group attributes be relativised to single beliefs? if so, 
        it makes it hard to capture how distrust spreads to other propositions.
        2. The truth + bias parameter should determine the aggregate judgement of experts on 
        every single proposition. This seems problematic. shouldn't it vary? But bias on 
        different propositions should have a strong correlation with each other too.
        '''