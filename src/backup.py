''' not sure if i need this. 
Might be helpful in simulating how to maximize accuracy. 
Accuracy is a relation between a credence and the truth.

"There are many different measures of accuracy that can be used; in this paper I will be assuming the
most standardly used measure, the Brier score, but as far as I am aware, assuming a different accuracy 
measure would not necessarily impact the argument." - https://link.springer.com/article/10.1007/s11229-020-02849-z
'''
class World():
    '''maps propositions to truth values'''
    def __init__(self) -> None:
        pass


# maybe not necessary
# agent's belief. this also stores the expert groups relevant for the belief
# class Credence():
#     def __init__(self, proposition, initial_credence = 0.5) -> None:
#         self.content = proposition
#         self.value = initial_credence
#         self.relevant_expert_groups = []

# class Expert_Group_Representation():
#     def __init__(self, ideal_credence_obj: Ideal_Credences, expert_group_obj: Expert_Group, estimated_population_bias_distributions, estimated_population_sd) -> None:
#         self.expert_group = expert_group_obj
#         self.ideal_credences = ideal_credence_obj
#         self.propositions = list(self.ideal_credences.keys())
#         self.estimated_population_bias_distributions = estimated_population_bias_distributions
#         self.estimated_population_sd = estimated_population_sd

#     def get_estimated_ideal_credences(self, agent_credences: dict):
#         return 