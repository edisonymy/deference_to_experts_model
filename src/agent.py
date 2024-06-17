import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils import name_generator
import scipy.stats as stats
import pymc as pm
import arviz as az
import corner
from typing import List
from plotting import plot_all_individual_judgements
from ideal_credences import Ideal_Credences
from experts import Expert_Group

class Bayesian_Agent():
    def __init__(self, ideal_credence_obj: Ideal_Credences) -> None:
        ''' should agent's priors on the bias of the expert group be a distribution? this allows us to adjust how flat the distribution is based on the agent's familiarity and confidence, which seems like a useful feature.'''
        self.priors = {} #keys are propositions, values are Credence objects
        # self.expert_group_representations = {} # deprecated? these are classes
        self.expert_group_models = {} # PYMC
        self.ideal_credence_models = {}
        self.ideal_credence_estimates = {}
        self.ideal_credences = ideal_credence_obj.credence_function
        self.propositions = list(self.ideal_credences.keys())    

    def get_expert_model(self, expert_group_obj: Expert_Group, prior_distribution = 'uniform', plot = True, credence_prior_mu = 0.5, credence_prior_sigma = 0.2, expert_bias_prior_mu = 0.5, expert_bias_prior_sigma = 0.2, expert_sd_prior_sigma = 0.2):
        ''' Sets up an expert model for each proposition using the specified prior distribution. '''
        self.expert_group_models[expert_group_obj] = {}
        # treats ideal credence as an unknown parameter and aggregate judgements as observed variable.
        for proposition in self.propositions:
            print('Generating model for proposition: ', proposition)
            pm_model = pm.Model()
            with pm_model:
                # set priors
                if prior_distribution == 'uniform':
                    proposition_credence = pm.Uniform(f'ideal_credence', 0, 1)
                    expert_bias = pm.Uniform(f'expert_bias', -1, 1)
                if prior_distribution == 'normal':
                    proposition_credence = pm.Normal(f'ideal_credence', mu= credence_prior_mu, sigma = credence_prior_sigma)
                    expert_bias = pm.Normal(f'expert_bias', mu = expert_bias_prior_mu, sigma = expert_bias_prior_sigma)
                expert_sd = pm.HalfNormal(f'expert_sd', sigma = expert_sd_prior_sigma)
                
                # define model
                # these are individual expert level, not group level, so each individual expert judgement is a separate observation
                expert_judgement = expert_bias + proposition_credence
                
                # sigma is the assumed noise parameter
                y_obs = pm.Normal('observed_judgements', mu = expert_judgement, sigma = expert_sd, observed = expert_group_obj.individual_judgements[proposition])

            self.expert_group_models[expert_group_obj][proposition] = pm_model
            if plot: 
                graph = pm.model_to_graphviz(pm_model)
                display(graph)
            self.priors[proposition] = proposition_credence

    def mcmc_get_posteriors(self, expert_group_obj: Expert_Group): # Expert_Group
        ''' Runs MCMC sampling to obtain posterior distributions for the expert group's propositions. '''
        results = []
        for proposition in self.propositions:
            pm_model = self.expert_group_models[expert_group_obj][proposition] 
            with pm_model:
                # draw 1000 posterior samples
                idata = pm.sample()
            results.append(idata)
            az.plot_trace(idata, combined=True)
            plt.tight_layout()
            corner_plot = corner.corner(
                idata,
                truths=dict(ideal_credence=self.ideal_credences[proposition], expert_bias=expert_group_obj.population_bias, expert_sd=expert_group_obj.population_sd),
            )
            # display(corner_plot)
        return results
    ### deprecated because PyMC ###

    # def get_priors(self, randomize_method = 'uniform'): 
    #     '''Not sure how to do this right. This shouldn't be thought of as just the priors from a blank slate. 
    #     Rather, this should reflect the credences that result from the agent's lived experiences, etc.
        
    #     Should this be a distribution rather than a scaler? we could use the flatness of the distribution to reflect uncertainty.This seems especially appropriate if the number is not interpreted as a credence but rather just some parameter the agent is estimating in general, e.g. the gravitational constant
    #     '''
    #     # creates a dictionary for self.credences
    #     if randomize_method == 'uniform':
    #         # for each proposition, set the credence to a random number between 0 and 1
    #         self.priors = {prop: pm.Uniform(f'{prop}_prior', lower=0, upper=1) for prop in self.propositions}

    # def get_agent_model(self, expert_group_obj: Expert_Group):
    #     for proposition, credence in self.credences.items():
    #         self.ideal_credence_models[proposition] = pm.Model()
    #         with self.ideal_credence_models[proposition]:
    #             #self.ideal_credence_estimates[proposition] = expert_group_obj.aggregate_judgements[proposition] - estimated_bias
    #             pass
        
    # def init_expert_group_representations(self, expert_groups: list, noise = 0.4):
    #     '''Not sure how to do this right.'''
    #     # creates a dictionary for self.expert_group_representations, with each key being an expert group, and each value being an expert group object that contains the agent's estimates on the expert group's attributes.
    #     # the initial estimates are randomly generated around the actual attributes, with the deviation determined by the noise parameter
    #     for expert_group in expert_groups:
    #         # random
    #         initial_population_bias = np.random.normal(loc=expert_group.population_bias, scale=noise) 
    #         initial_population_sd = np.random.normal(loc=expert_group.population_sd, scale=noise)

    #         self.expert_group_representations[expert_group] = Expert_Group(ideal_credences=self.ideal_credences, population_bias=initial_population_bias, population_sd=initial_population_sd, simulate = False)
        


    # def update_expert_group_representations(self, expert_groups: list):
    #     # updates all expert group representations at once. since, for bayesian agents, order shouldn't matter. even if updates are sequential, they should apply retroactively too. 
    #     # how expert group representations get updated should be a function of priors, and expert group representations from previous update
    #     # should generate a one to one mapping from each expert group to an expert group representation, with estimates for the attributes. 
    #     # fuck, the representations have to be probability distributions. since the result of the update should be a probability distribution, the inputs might as well be.
        
    #     for expert_group in expert_groups:
    #         # this is a rough sketch. not sure if this is correct
    #         estimated_population_bias_distributions = {prop: aggregate_judgement - self.credences[prop] for prop, aggregate_judgement in expert_group.aggregate_judgements.items()}
    #         estimated_population_sd = {prop: sample_sd for prop, sample_sd in expert_group.sample_sd_dict.items()}
    #         self.expert_group_representations[expert_group] = Expert_Group_Representation(self.ideal_credences, expert_group, estimated_population_bias_distributions, estimated_population_sd)
    
    
    # def get_posterior_distribition(self, propositions, expert_groups: list):
    #     # this is the agent's credence distribution of the ideal credence which should be a function of three inputs: the expert sample distributions, the agent's priors on the respective propositions
    #     pass


    # def update_proposition_credence(self, propositions, expert_groups: list):
    #     # this is the agent's estimate of the ideal credence which should be a function of three inputs: the expert sample distributions, the agent's priors on the respective propositions
    #     # maybe this is derived from the posterior distribution, just the point that minimizes expected error?
    #     pass