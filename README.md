# Probabilistic Finite Automata Based Text Generator

> **Theory of Computation Mini Project**

## Overview

This project demonstrates how a **Probabilistic Finite Automaton (PFA)** can be used
to generate text using concepts from Theory of Computation.

### Core Idea

| ToC Concept       | Implementation                                       |
|-------------------|------------------------------------------------------|
| **States**        | Bigrams — every pair of consecutive words            |
| **Alphabet**      | All unique words in the training text                |
| **Transitions**   | Probability of the next word given the current state |
| **Start State**   | A randomly chosen bigram from the training text      |
| **Dead State**    | A bigram with no outgoing transitions (end of chain) |

The model is essentially a **first-order Markov chain** where the state is a bigram
(two words), making it a second-order language model.

---

## Project Structure

```
toc/
├── app.py              # Streamlit UI
├── markov_pfa.py       # Core PFA / Markov chain logic
├── sample_corpus.txt   # Sample training text
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Features

1. **Train** — Paste or type any text into the training area and click **Train Model**.
2. **Inspect** — View the full PFA transition table (state → next word → probability).
3. **Filter** — Search the transition table by state name.
4. **Generate** — Choose how many words to generate and click **Generate Text**.
5. **Trace** — Expand the *State Transition Path* to see every automaton step taken during generation.

---

## How the Code Works

### `markov_pfa.py`

| Function                  | What it does                                                  |
|---------------------------|---------------------------------------------------------------|
| `tokenize(text)`          | Cleans and splits raw text into a list of lowercase words     |
| `build_pfa(tokens)`       | Counts bigram → next-word occurrences and converts to probs   |
| `generate_text(pfa, n)`   | Walks the PFA probabilistically to produce `n` words         |
| `build_transition_table(pfa)` | Flattens the PFA dict into rows for display             |

### `app.py`

Streamlit widgets used:
- `st.text_area` — training text input
- `st.button` — Train / Generate actions
- `st.dataframe` — transition table display
- `st.number_input` — word count selector
- `st.session_state` — persists the trained model across button clicks

---

## Example

**Training text:**  
`"the cat sat on the mat the cat ate the rat"`

**Learned states (bigrams):**  
`(the, cat)`, `(cat, sat)`, `(sat, on)`, `(on, the)`, …

**Sample transition:**  
`(the, cat)` → `sat` with prob **0.5**, `ate` with prob **0.5**

**Generated output:**  
`"the cat sat on the mat the cat ate the rat"`

---

## Requirements

- Python 3.8+
- streamlit
- pandas
- streamlit-keyup
