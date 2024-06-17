import matplotlib.pyplot as plt
import seaborn as sns
def plot_all_individual_judgements(expert_groups):
    """
    Plots the individual judgements of multiple expert groups on the same histogram for each proposition.

    Parameters:
    - expert_groups: A list of Expert_Group instances.
    """
    if not expert_groups:
        print("No expert groups provided.")
        return
    
    # Define a list of colors or use a colormap
    colors = sns.color_palette('hsv', len(expert_groups))

    # Assuming all expert groups have the same propositions
    propositions = list(expert_groups[0].propositions)
    n_props = len(propositions)
    plt.figure(figsize=(14, 4 * n_props + 2))

    for i, prop in enumerate(propositions, start=1):
        plt.subplot(n_props, 1, i)
        for idx, expert_group in enumerate(expert_groups):
            # Use the same color for individual judgements and aggregate judgement line of the expert group
            group_color = colors[idx]
            label = f'(Bias: {expert_group.population_bias}, population sd: {expert_group.population_sd:.3f}, sample mean sd: {expert_group.sample_mean_sd:.3f}, sample sd: {expert_group.sample_sd_dict[prop]:.3f})'
            #label = f'{expert_group.name} (Bias: {expert_group.bias}, Consensus: {expert_group.consensus}, Competence: {expert_group.competence})'
            sns.histplot(expert_group.individual_judgements[prop], kde=False, bins='auto', color=group_color, label=label, stat="density")
            plt.axvline(x=expert_group.aggregate_judgements[prop], color=group_color, linestyle='--', label=f'{expert_group.name} aggregate judgement')
        
        # Ideal credence line
        plt.axvline(x=expert_groups[0].ideal_credences[prop], color='black', linestyle='--', label=f'Ideal credence for "{prop}"')
        plt.legend()
        plt.title(f'Individual Judgements Distribution for "{prop}"')
        plt.xlabel('Value')
        plt.xlim(0,1)
        plt.ylim(0,15)
        plt.ylabel('Density')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2, title='')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()