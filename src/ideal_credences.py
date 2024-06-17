import random
from utils import name_generator
from typing import List
from plotting import plot_all_individual_judgements
class Ideal_Credences():
    ''' 
    Maps propositions to credences

    truth of a proposition is binary but ideal credence can range from 0 to 1. 
    Ideal credence is the credence that best fulfills epistemic norms given all 
    available evidence (not necessarily accessible to you).

    "I will assume that credences are sharp and
    that the uniqueness thesis is true (Elga 2010; Kopec & Titelbaum 2016). In other
    words, I will assume that for any proposition p and any set of evidence E there is
    only one maximally rational credence that a perfectly rational epistemic agent could
    have, which is a sharp value and not an interval, and any deviation from the unique
    maximally rational credence constitutes a deviation from rationality."
     - https://link.springer.com/article/10.1007/s11229-020-02849-z
    '''
    # the odds you would take on bet on is sharp. 
    def __init__(self, propositions: list, seed = None) -> None:
        ''' Initializes the Ideal_Credences class with a list of propositions and an optional seed for randomness. '''
        if seed is not None:
            random.seed(seed)
        self.credence_function = {}
        self.propositions = propositions
        

    def random_generate_ideal_credences(self):
        ''' Randomly generates ideal credences for the propositions, with values ranging between 0 and 1. '''
        self.credence_function = {proposition: random.uniform(0, 1) for proposition in self.propositions}
        
