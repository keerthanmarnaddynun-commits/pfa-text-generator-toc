"""
markov_pfa.py
-------------
Core logic for the Probabilistic Finite Automaton (PFA) based text generator.

A PFA is like a regular finite automaton, but transitions are labeled with
probabilities instead of certainties. Here, each STATE is a bigram (pair of
consecutive words), and each TRANSITION is the probability of moving to a
particular next word given the current state.

Theory of Computation connection:
  - States     -> unique bigrams (pairs of words)
  - Alphabet   -> all unique words in the corpus
  - Transitions-> probability distribution over next words from each state
  - Start state-> first bigram in the text
  - Accept/end -> states with no outgoing transitions
"""

import random
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Convert raw text into a list of lowercase words.
    Strips punctuation so that 'hello.' and 'hello' are treated as the same.
    """
    # Replace anything that is not a letter, digit, or whitespace with a space
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Split on whitespace and lowercase every token
    tokens = cleaned.lower().split()
    return tokens


# ---------------------------------------------------------------------------
# Building the PFA (Markov chain with bigram states)
# ---------------------------------------------------------------------------

def build_pfa(tokens: list) -> dict:
    """
    Build a Probabilistic Finite Automaton from a list of tokens.

    Returns a nested dict:
        pfa[state] = { next_word: probability, ... }

    where `state` is a tuple (word1, word2) representing a bigram.

    Steps:
        1. Count how many times each next_word follows each bigram state.
        2. Divide each count by the total to get probabilities.
    """

    # Step 1: Count transitions
    # raw_counts[state] = { next_word: count }
    raw_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - 2):          # need at least 3 tokens
        state = (tokens[i], tokens[i + 1])    # bigram state
        next_word = tokens[i + 2]             # word that follows
        raw_counts[state][next_word] += 1

    # Step 2: Convert counts to probabilities
    pfa = {}
    for state, next_words in raw_counts.items():
        total = sum(next_words.values())      # total transitions from this state
        pfa[state] = {
            word: count / total
            for word, count in next_words.items()
        }

    return pfa


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(pfa: dict, num_words: int, seed: int = None) -> tuple:
    """
    Generate text by walking the PFA probabilistically.

    Args:
        pfa      : The PFA dict returned by build_pfa().
        num_words: Approximate number of words to generate.
        seed     : Optional random seed for reproducibility.

    Returns:
        (generated_text, path)
        - generated_text : The output string.
        - path           : List of (state, chosen_next_word) tuples showing
                           the exact sequence of state transitions taken.
    """

    if not pfa:
        return "", []

    if seed is not None:
        random.seed(seed)

    # Pick a random starting state from all available states
    all_states = list(pfa.keys())
    current_state = random.choice(all_states)

    # The generated word list starts with the two words in the starting state
    words = list(current_state)

    # path records each transition: (state, next_word)
    path = []

    # Walk the PFA until we have enough words or hit a dead end
    for _ in range(num_words - 2):            # -2 because start state gives 2 words
        if current_state not in pfa:
            break                             # dead end — no outgoing transitions

        transitions = pfa[current_state]

        # Weighted random choice based on transition probabilities
        next_word = random.choices(
            population=list(transitions.keys()),
            weights=list(transitions.values()),
            k=1
        )[0]

        # Record the transition we just took
        path.append((current_state, next_word))

        # Append the chosen word and advance to the next state
        words.append(next_word)
        current_state = (current_state[1], next_word)   # slide the bigram window

    generated_text = " ".join(words)
    return generated_text, path


# ---------------------------------------------------------------------------
# Building the transition table (for display)
# ---------------------------------------------------------------------------

def build_transition_table(pfa: dict) -> list:
    """
    Flatten the nested PFA dict into a list of row-dicts suitable for a
    pandas DataFrame or Streamlit table.

    Each row has:
        - "Current State" : string representation of the bigram, e.g. "(the, cat)"
        - "Next Word"     : the word being transitioned to
        - "Probability"   : rounded probability value
    """
    rows = []
    for state, transitions in pfa.items():
        state_str = f"({state[0]}, {state[1]})"
        for next_word, prob in transitions.items():
            rows.append({
                "Current State": state_str,
                "Next Word": next_word,
                "Probability": round(prob, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# PFA Statistics (used by the Statistics Panel in the UI)
# ---------------------------------------------------------------------------

def get_pfa_stats(tokens: list, pfa: dict) -> dict:
    """
    Compute summary statistics about the trained PFA.

    Returns a dict with:
        - total_tokens     : number of words in the training text
        - total_states     : number of unique bigram states
        - total_transitions: total (state, next-word) pairs across all states
        - avg_branching    : average number of outgoing transitions per state
                             (= average number of next-word choices from any state)

    The 'branching factor' is an important concept in automata theory.
    A DFA has a branching factor of exactly 1 (deterministic).
    A PFA has a branching factor >= 1 (probabilistic / non-deterministic).
    """
    total_states = len(pfa)
    # Count how many next-word options each state has, then sum them up
    transition_counts = [len(nexts) for nexts in pfa.values()]
    total_transitions = sum(transition_counts)
    avg_branching = (total_transitions / total_states) if total_states > 0 else 0.0

    return {
        "total_tokens": len(tokens),
        "total_states": total_states,
        "total_transitions": total_transitions,
        "avg_branching": round(avg_branching, 2),
    }


# ---------------------------------------------------------------------------
# Autocomplete / Next-Word Suggestion
# ---------------------------------------------------------------------------

def get_next_word_suggestions(pfa: dict, user_input: str, top_n: int = 5) -> tuple:
    """
    Simulates autocomplete by looking up the last two words of the user_input
    as a bigram state in the PFA, and returning the most likely next words.

    Args:
        pfa: The trained PFA.
        user_input: The input string.
        top_n: Maximum number of suggestions to return.

    Returns:
        (suggestions, current_state, message)
        - suggestions: List of (word, probability) tuples, or empty list.
        - current_state: Tuple of two words used as the bigram state, or None.
        - message: A status message (warning/info/error) or None.
    """
    # 1. Tokenize user input using the standard tokenizer to ensure consistency
    tokens = tokenize(user_input)

    # 2. Check if we have at least 2 tokens (bigram state requires two words)
    if len(tokens) < 2:
        return [], None, "Enter at least two words because this model uses bigram states."

    # 3. Take the last two tokens as the current state (sliding window of size 2)
    current_state = (tokens[-2], tokens[-1])

    # 4. If the state is not in our PFA, we cannot make any suggestions
    if current_state not in pfa:
        return [], current_state, "This state was not found in the trained corpus. Try words from the sample corpus."

    # 5. Retrieve all possible next words and their learned probabilities
    transitions = pfa[current_state]

    # 6. Sort transitions by probability descending.
    # We also sort by word alphabetically to ensure stable tie-breaking when probabilities are equal.
    sorted_suggestions = sorted(
        transitions.items(),
        key=lambda item: (-item[1], item[0])
    )

    return sorted_suggestions[:top_n], current_state, None

