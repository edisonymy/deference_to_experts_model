import random
def name_generator():
    adjectives = ['Strategic', 'Innovative', 'Analytical', 'Dynamic', 'Global']
    nouns = ['Thinkers', 'Pioneers', 'Strategists', 'Analysts', 'Leaders']

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{adjective.capitalize()} {noun.capitalize()}"