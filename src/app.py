import streamlit as st
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.stats import beta, norm, uniform
from ideal_credences import Ideal_Credences
from experts import Expert_Group
from bayesian_agent import Bayesian_Agent
import graphviz

st.set_page_config(layout="wide")

st.title("Bayesian Updating with Expert Testimony")

st.write("""
This app simulates how a Bayesian agent updates their beliefs based on expert testimony.
You can set your prior beliefs about specific propositions, define expert groups, and see how the beliefs change after 
considering the experts' opinions.
""")

# Sidebar for input parameters
st.sidebar.header("Simulation Parameters")

# Beliefs setup
st.sidebar.subheader("1. Define Your Beliefs")
num_propositions = st.sidebar.number_input("Number of Propositions", 1, 5, 2, 
                                   help="How many different statements or ideas do you want to consider?")

# Allow users to enter custom propositions
proposition_descriptions = []
for i in range(num_propositions):
    default_prop = f"The global average temperature will rise by more than 2°C by 2050 (Proposition {i+1})"
    prop = st.sidebar.text_input(f"Proposition {i+1}", value=default_prop, 
                                 help="Enter a statement you want to assess")
    proposition_descriptions.append(prop)

prior_distribution = st.sidebar.selectbox("Prior Distribution Type", ["uniform", "normal", "beta"],
                                  help="What kind of distribution best represents your prior beliefs?")

# Allow users to set their priors
priors = {}
if prior_distribution != "uniform":
    st.sidebar.write("Set Your Priors for the Propositions")
    for i, prop_desc in enumerate(proposition_descriptions):
        st.sidebar.write(f"Proposition {i+1}: {prop_desc}")
        if prior_distribution == "normal":
            mean = st.sidebar.slider(f"Mean for Proposition {i+1}", 0.0, 1.0, 0.5, 0.01,
                             help="The average probability you assign to this proposition being true")
            std = st.sidebar.slider(f"Standard Deviation for Proposition {i+1}", 0.01, 0.5, 0.2, 0.01,
                            help="How uncertain you are about this probability")
            priors[f'p{i+1}'] = {'mean': mean, 'std': std}
        elif prior_distribution == "beta":
            alpha = st.sidebar.slider(f"Alpha for Proposition {i+1}", 0.1, 10.0, 2.0, 0.1,
                              help="Shape parameter for the beta distribution (higher values push the distribution to the right)")
            beta_param = st.sidebar.slider(f"Beta for Proposition {i+1}", 0.1, 10.0, 2.0, 0.1,
                                   help="Shape parameter for the beta distribution (higher values push the distribution to the left)")
            priors[f'p{i+1}'] = {'alpha': alpha, 'beta': beta_param}

# Expert Groups setup
st.sidebar.subheader("2. Define Expert Groups")
num_expert_groups = st.sidebar.number_input("Number of Expert Groups", 1, 5, 2,
                                    help="How many different groups of experts do you want to include?")

expert_groups = []
for i in range(num_expert_groups):
    st.sidebar.write(f"Expert Group {i+1}")
    group_name = st.sidebar.text_input(f"Name for Expert Group {i+1}", value=f"Group {i+1}",
                                       help="Give a name to this group of experts (e.g., 'Climate Scientists', 'Economists')")
    n_experts = st.sidebar.number_input(f"Number of Experts in {group_name}", 10, 200, 100,
                                help="How many experts are in this group?")
    population_sd = st.sidebar.slider(f"Population Standard Deviation ({group_name})", 0.0, 1.0, 0.2, 0.1,
                              help="How much do the experts in this group tend to disagree with each other?")
    population_bias = st.sidebar.slider(f"Population Bias ({group_name})", -1.0, 1.0, 0.1, 0.1,
                                help="Does this group tend to overestimate or underestimate? (0 means no bias)")
    expert_groups.append({"name": group_name, "n_experts": n_experts, "population_sd": population_sd, "population_bias": population_bias})

# MCMC parameters in collapsible section
st.sidebar.subheader("3. Advanced Settings")
with st.sidebar.expander("MCMC Parameters (for advanced users)"):
    draws = st.slider("Number of Draws", 100, 5000, 1000, 
                      help="More draws give more accurate results but take longer to compute")
    tune = st.slider("Tuning Steps", 100, 2000, 500,
                     help="Steps used to tune the MCMC algorithm before drawing samples")
    chains = st.slider("Number of Chains", 1, 4, 2,
                       help="Independent MCMC chains to run")
    target_accept = st.slider("Target Accept Rate", 0.5, 1.0, 0.95, 0.05,
                              help="Target acceptance rate for the MCMC algorithm")

# Run Simulation button in the main body
if st.button("Run Simulation"):
    # Initialize ideal credences
    ideal_credences = Ideal_Credences(propositions=[f'p{i+1}' for i in range(num_propositions)])
    ideal_credences.random_generate_ideal_credences()

    # Initialize expert groups
    expert_group_objects = []
    for group in expert_groups:
        expert_group = Expert_Group(ideal_credences, name=group['name'], n_experts=group['n_experts'], 
                                    population_sd=group['population_sd'], population_bias=group['population_bias'])
        expert_group_objects.append(expert_group)

    # Initialize Bayesian agent
    bayesian_agent = Bayesian_Agent(ideal_credences)

    # Generate hierarchical model for multiple groups
    with st.spinner("Generating hierarchical model..."):
        bayesian_agent.get_hierarchical_model(expert_groups=expert_group_objects, 
                                              prior_distribution=prior_distribution, 
                                              priors=priors,
                                              plot=False)
        
        # Generate and display the graphviz model
        graph = pm.model_to_graphviz(bayesian_agent.hierarchical_model)
        st.graphviz_chart(graph)

    # Run MCMC to get posteriors for the hierarchical model
    with st.spinner("Running MCMC simulation..."):
        progress_bar = st.progress(0)
        results = bayesian_agent.mcmc_get_posteriors(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        progress_bar.progress(100)

    st.success("Simulation completed!")

    # Plot results
    st.header("Simulation Results")

    # Plot ideal credences
    fig_credences, axes_credences = plt.subplots(1, 3, figsize=(20, 5))
    for i, prop in enumerate(ideal_credences.credence_function.keys()):
        # Prior
        x = np.linspace(0, 1, 1000)
        if prior_distribution == 'uniform':
            y = uniform.pdf(x, loc=0, scale=1)
        elif prior_distribution == 'normal':
            mean, std = priors[prop]['mean'], priors[prop]['std']
            y = norm.pdf(x, loc=mean, scale=std)
        elif prior_distribution == 'beta':
            alpha, beta_param = priors[prop]['alpha'], priors[prop]['beta']
            y = beta.pdf(x, a=alpha, b=beta_param)
        axes_credences[0].plot(x, y, label=f'{proposition_descriptions[i]}')
        
        # Posterior
        az.plot_posterior(results, var_names=['ideal_credences'], coords={'ideal_credences_dim_0': [i]}, ax=axes_credences[1])
        
        # Truth
        axes_credences[2].axvline(ideal_credences.credence_function[prop], color=f'C{i}', linestyle='--', label=f'{proposition_descriptions[i]}')

    axes_credences[0].set_title("Your Prior Beliefs")
    axes_credences[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes_credences[1].set_title("Updated Beliefs (Posterior)")
    axes_credences[2].set_title("True Values")
    axes_credences[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_credences)

    # Plot expert biases and SDs
    fig_experts, axes_experts = plt.subplots(2, 3, figsize=(20, 10))
    for i, expert_group in enumerate(expert_group_objects):
        # Biases
        x = np.linspace(-1, 2, 1000)
        y = norm.pdf(x, loc=0.5, scale=0.2)
        axes_experts[0, 0].plot(x, y, label=f'{expert_group.name}')
        az.plot_posterior(results, var_names=['expert_biases'], coords={'expert_biases_dim_0': [i]}, ax=axes_experts[0, 1])
        axes_experts[0, 2].axvline(expert_group.population_bias, color=f'C{i}', linestyle='--', label=f'{expert_group.name}')

        # SDs
        x = np.linspace(0, 1, 1000)
        y = (2/(np.pi*0.2**2))**0.5 * np.exp(-x**2/(2*0.2**2))
        axes_experts[1, 0].plot(x, y, label=f'{expert_group.name}')
        az.plot_posterior(results, var_names=['expert_sds'], coords={'expert_sds_dim_0': [i]}, ax=axes_experts[1, 1])
        axes_experts[1, 2].axvline(expert_group.population_sd, color=f'C{i}', linestyle='--', label=f'{expert_group.name}')

    axes_experts[0, 0].set_title("Prior Expert Biases")
    axes_experts[0, 0].legend()
    axes_experts[0, 1].set_title("Estimated Expert Biases")
    axes_experts[0, 2].set_title("True Expert Biases")
    axes_experts[0, 2].legend()

    axes_experts[1, 0].set_title("Prior Expert Standard Deviations")
    axes_experts[1, 0].legend()
    axes_experts[1, 1].set_title("Estimated Expert Standard Deviations")
    axes_experts[1, 2].set_title("True Expert Standard Deviations")
    axes_experts[1, 2].legend()
    plt.tight_layout()
    st.pyplot(fig_experts)

else:
    st.info("Adjust the parameters in the sidebar, then click 'Run Simulation' to start.")