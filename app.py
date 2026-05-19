"""
app.py
------
Streamlit UI for the Probabilistic Finite Automata Based Text Generator.

Theory of Computation perspective:
  - Training builds a PFA where:
      * States      = bigrams (pairs of consecutive words)
      * Alphabet    = words in the vocabulary
      * Transitions = probability distributions over next words
  - Text generation = a probabilistic walk through the automaton states.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
try:
    from streamlit_keyup import st_keyup
except ModuleNotFoundError:
    from st_keyup import st_keyup


# graphviz is used to draw the automata diagram.
# It is a pure-Python package — no separate system install needed for basic graphs.
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# Import our core PFA logic from markov_pfa.py
from markov_pfa import (
    tokenize,
    build_pfa,
    generate_text,
    build_transition_table,
    get_pfa_stats,          # NEW — statistics helper
    get_next_word_suggestions,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Probabilistic Finite Automata Based Text Generator | Theory of Computation",
    page_icon="🤖",
    layout="wide",
)

if "training_text_backup" not in st.session_state:
    st.session_state.training_text_backup = ""

# Callback to load the sample corpus directly into session_state before rerun
def load_sample_callback():
    try:
        with open("sample_corpus.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "Could not find sample_corpus.txt. Please type your training text here."
    st.session_state.training_text_backup = text
    st.session_state.training_text_widget = text





with st.sidebar:
    st.title("📘 Project Info")
    st.markdown(
        """
        **Course:** Theory of Computation

        **Topic:** Probabilistic Finite Automata
        """
    )
    st.header("🗺️ Navigation")
    page = st.radio(
        "Navigate",
        ["Introduction", "PFA Training & Visualization", "Next-Word Suggestion"],
        index=0,
        key="navigation_page"
    )

    st.header("⚙️ Quick Options")
    st.button("📄 Load Sample Corpus", on_click=load_sample_callback)
    st.markdown("---")
    st.caption("Theory of Computation")


# ---------------------------------------------------------------------------
# Page Routing Block (Clean if/elif/else structure)
# ---------------------------------------------------------------------------
if page == "Introduction":
    st.title("🤖 Probabilistic Finite Automata — Text Generator")
    st.markdown(
        "##### Theory of Computation &nbsp;|&nbsp; Bigram Markov Chain as a PFA"
    )
    st.markdown("---")
    
    st.markdown("### 📋 Project Overview")
    st.markdown(
        "This project demonstrates a Markov-chain based Probabilistic Finite Automaton for text generation and next-word suggestion."
    )
    st.markdown("---")

    st.markdown("### ❓ What is a PFA?")
    st.markdown(
        """
        A Probabilistic Finite Automaton is an extension of a regular Finite Automaton (DFA/NFA) where each transition carries a probability instead of being deterministic.

        In this project:
        - State = a bigram (two consecutive words)
        - Transition = probability of the next word
        - Walk = the generated sentence
        """
    )
    st.markdown("---")

    # Theory of Computation Mapping
    with st.expander("📖 Theory of Computation Mapping  *(click to expand)*", expanded=True):
        st.markdown(
            """
            ### How this project maps to Automata Theory

            | TOC Concept | This Project |
            |---|---|
            | **State (Q)** | A bigram — two consecutive words, e.g. `(the, cat)` |
            | **Alphabet (Σ)** | All unique words in the training text |
            | **Transition function (δ)** | Probability distribution over next words for each state |
            | **Start state (q₀)** | A randomly selected bigram from the training text |
            | **Dead / Accept state** | A bigram that appears only at the end (no outgoing transitions) |
            | **Input string** | The sequence of words read during text generation |
            | **Accepted string** | The generated sentence = one possible automaton traversal |

            ---
            ### Key ideas explained simply

            **1. Each bigram = one state**
            > If the sentence is *"the cat sat on the mat"*, then `(the, cat)`, `(cat, sat)`,
            > `(sat, on)` etc. are all distinct states in our automaton.

            **2. Each next-word prediction = a transition**
            > From state `(the, cat)`, the automaton might go to `sat` or `ate`.
            > The probability of each choice is learned from the training text.

            **3. Probabilities simulate PFA behavior**
            > A Deterministic FA (DFA) has exactly one arrow from each state.
            > A PFA has *multiple* arrows, each with a probability that sums to 1.
            > Our model works exactly like a PFA — the "random walk" picks an arrow
            > based on its weight.

            **4. The generated sentence = an automaton traversal path**
            > Every time you click *Generate*, the automaton starts in a random state
            > and follows transitions probabilistically until it produces the requested
            > number of words. The full path of states visited is shown in the
            > *State Transition Path* expander below the output.
            """
        )

    st.markdown("---")

    # DFA vs PFA Comparison
    with st.expander("⚖️ DFA vs Probabilistic Automata Comparison  *(click to expand)*", expanded=True):
        st.markdown("### How does a PFA differ from a DFA?")
        comparison_data = {
            "Property": [
                "Full name",
                "Transitions per state",
                "Transition type",
                "Next state",
                "Output",
                "Use case",
                "This project",
            ],
            "DFA (Deterministic FA)": [
                "Deterministic Finite Automaton",
                "Exactly 1 per input symbol",
                "Certain (probability = 1.0)",
                "Always the same",
                "Accept or Reject",
                "Pattern matching, compilers",
                "❌ Not used",
            ],
            "PFA (Probabilistic FA)": [
                "Probabilistic Finite Automaton",
                "Multiple (one per possible next word)",
                "Weighted (probabilities sum to 1)",
                "Randomly chosen based on probability",
                "A generated sequence",
                "Language modeling, text generation",
                "✅ This is what we build",
            ],
        }
        st.table(pd.DataFrame(comparison_data))

elif page == "PFA Training & Visualization":
    st.title("🤖 Probabilistic Finite Automata — PFA Training & Visualization")
    st.markdown(
        "##### Train the Markov chain model and visualize the resulting transition graph."
    )
    st.markdown("---")

    # Step 1 — Train the PFA
    st.header("Step 1 — Train the PFA")
    st.markdown(
        "Paste any text below and click **Train Model**. "
        "The app will tokenize the text, extract all bigram states, "
        "and compute transition probabilities."
    )

    # Text area — user types or pastes training text here
    training_text = st.text_area(
        label="Enter training text below:",
        value=st.session_state.training_text_backup,
        height=200,
        placeholder="Paste or type training text here...",
        help="The model will learn word transition probabilities from this text.",
        key="training_text_widget"
    )

    st.session_state.training_text_backup = training_text

    # Train button
    train_btn = st.button("🔧 Train Model", type="primary")

    # When the user clicks Train, build the PFA and store it in session_state
    if train_btn:
        training_text = st.session_state.training_text_backup
        if not training_text.strip():
            st.error("Please enter some training text before training the model.")
        else:
            tokens = tokenize(training_text)

            if len(tokens) < 3:
                st.error("Text is too short. Please provide at least 3 words.")
            else:
                st.session_state.tokens = tokens
                st.session_state.pfa = build_pfa(tokens)
                st.session_state.trained = True

                st.success(
                    f"✅ Model trained!  "
                    f"**{len(tokens)}** tokens found.  "
                    f"**{len(st.session_state.pfa)}** unique bigram states created."
                )

    if st.session_state.get("trained", False):
        # PFA Statistics
        st.markdown("---")
        st.header("📊 PFA Statistics")
        st.markdown(
            "A quick overview of the automaton that was built from your training text."
        )

        stats = get_pfa_stats(st.session_state.tokens, st.session_state.pfa)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(
            label="🔤 Total Tokens",
            value=stats["total_tokens"],
            help="Total number of words in the training text after tokenization.",
        )
        col_b.metric(
            label="🔵 Total States",
            value=stats["total_states"],
            help="Number of unique bigram states (nodes) in the PFA.",
        )
        col_c.metric(
            label="➡️ Total Transitions",
            value=stats["total_transitions"],
            help="Total number of (state → next-word) arrows in the automaton.",
        )
        col_d.metric(
            label="🌿 Avg Branching Factor",
            value=stats["avg_branching"],
            help=(
                "Average number of outgoing transitions per state. "
                "A DFA always has branching factor = 1. "
                "Higher values mean more non-determinism."
            ),
        )

        # Step 2 — Transition table
        st.markdown("---")
        st.header("Step 2 — PFA Transition Table")
        st.markdown(
            """
            The table below shows every **state → next-word** transition along
            with its probability. Each row represents one **arrow** in the automaton diagram.

            > 💡 In a DFA, each state would have exactly **one** row.
            > In our PFA, a state can have **many** rows — one per possible next word.
            """
        )

        table_rows = build_transition_table(st.session_state.pfa)

        if table_rows:
            df = pd.DataFrame(table_rows)

            filter_text = st.text_input(
                "🔍 Filter by state (optional):",
                value="",
                placeholder="e.g.  the, cat",
                help="Type part of a state name to filter the table.",
            )

            if filter_text.strip():
                mask = df["Current State"].str.contains(filter_text.strip(), case=False, na=False)
                df_display = df[mask]
            else:
                df_display = df

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.caption(
                f"Total transitions in the PFA: **{len(table_rows)}**  |  "
                f"Showing: **{len(df_display)}**"
            )
        else:
            st.warning("No transitions found. Try with more text.")

        # Automata Visualization
        st.markdown("---")
        st.header("🔗 Automata Visualization")
        st.markdown(
            """
            The directed graph below shows a **subset** of the PFA states and transitions.
            - Each **node (circle)** = a bigram state
            - Each **arrow** = a transition to the next word
            - The **label** on each arrow = the transition probability

            > Only the first few states are shown to keep the graph readable.
            > Use the transition table above for the full list.
            """
        )

        pfa_size = len(st.session_state.pfa)
        if pfa_size <= 3:
            MAX_STATES_TO_SHOW = pfa_size
        else:
            MAX_STATES_TO_SHOW = st.slider(
                "Number of states to display in the graph:",
                min_value=3,
                max_value=min(20, pfa_size),
                value=min(8, pfa_size),
                step=1,
                help="More states = larger graph. Keep it small for readability.",
            )

        if GRAPHVIZ_AVAILABLE:
            dot = graphviz.Digraph(
                name="PFA",
                comment="Probabilistic Finite Automaton",
                graph_attr={
                    "rankdir": "LR",
                    "fontsize": "11",
                    "bgcolor": "white",
                    "splines": "curved",
                },
                node_attr={
                    "shape": "ellipse",
                    "style": "filled",
                    "fillcolor": "#e8f4f8",
                    "fontsize": "10",
                    "fontname": "Helvetica",
                },
                edge_attr={
                    "fontsize": "9",
                    "fontname": "Helvetica",
                    "color": "#555555",
                },
            )

            pfa_sample = dict(list(st.session_state.pfa.items())[:MAX_STATES_TO_SHOW])

            for state in pfa_sample:
                state_label = f"{state[0]}, {state[1]}"
                dot.node(
                    name=str(state),
                    label=state_label,
                )

            for state, transitions in pfa_sample.items():
                for next_word, prob in transitions.items():
                    next_state = (state[1], next_word)
                    next_label = f"{state[1]}, {next_word}"

                    if next_state not in pfa_sample:
                        dot.node(
                            name=str(next_state),
                            label=next_label,
                            fillcolor="#fff3cd",
                            style="filled,dashed",
                        )

                    dot.edge(
                        tail_name=str(state),
                        head_name=str(next_state),
                        label=f" {prob:.2f}",
                    )

            st.graphviz_chart(dot, use_container_width=True)
            st.caption(
                "🟦 Blue nodes = states fully inside the sample window.  "
                "🟨 Yellow dashed nodes = states outside the sample window (destination only)."
            )
        else:
            st.warning(
                "The `graphviz` Python package is not installed. "
                "Run `pip install graphviz` then restart the app to see the graph."
            )

        # Step 3 — Generate text
        st.markdown("---")
        st.header("Step 3 — Generate Text  (Automaton Walk)")
        st.markdown(
            "Click the button to start a **probabilistic walk** through the PFA. "
            "The automaton picks a random starting state and follows weighted transitions "
            "until the requested number of words is reached."
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            num_words = st.number_input(
                "Number of words to generate:",
                min_value=5,
                max_value=300,
                value=30,
                step=5,
                help="How many words should the generated text contain?",
            )

        with col2:
            use_seed = st.checkbox("Use fixed random seed (for reproducibility)", value=False)
            seed_value = None
            if use_seed:
                seed_value = st.number_input("Seed value:", min_value=0, value=42, step=1)

        generate_btn = st.button("✨ Generate Text", type="primary")

        if generate_btn:
            generated, path = generate_text(
                pfa=st.session_state.pfa,
                num_words=int(num_words),
                seed=int(seed_value) if seed_value is not None else None,
            )
            st.session_state.generated_text = generated
            st.session_state.generated_path = path

        if st.session_state.get("generated_text"):
            st.subheader("📝 Generated Text")
            st.success(st.session_state.generated_text)

            if st.session_state.get("generated_path"):
                with st.expander("🔁 Show State Transition Path  (automaton walk step-by-step)"):
                    st.markdown(
                        """
                        Each row below shows one **step** of the automaton walk:
                        - **Current State** = the bigram the automaton is currently in
                        - **Chosen Next Word** = the word it transitioned to (chosen probabilistically)

                        This sequence of states is the **computation path** of the PFA
                        on the generated string.
                        """
                    )
                    path_rows = [
                        {
                            "Step": idx + 1,
                            "Current State (q)": f"({s[0]}, {s[1]})",
                            "Chosen Next Word (σ)": word,
                            "Next State (δ(q,σ))": f"({s[1]}, {word})",
                        }
                        for idx, (s, word) in enumerate(st.session_state.generated_path)
                    ]
                    path_df = pd.DataFrame(path_rows)
                    st.dataframe(path_df, use_container_width=True, hide_index=True)

elif page == "Next-Word Suggestion":
    st.title("🔮 Next-Word Suggestion System")
    st.markdown(
        "##### Autocomplete Simulator &nbsp;|&nbsp; Bigram State Transition Prediction"
    )
    st.markdown("---")
    st.markdown(
        "This simulates autocomplete using the trained probabilistic finite automaton. "
        "It reads your input, extracts the last two words as the current bigram state, "
        "and shows the most likely next transitions."
    )

    if not st.session_state.get("trained", False):
        st.warning("⚠️ Please train the PFA model first from the **PFA Training & Visualization** page.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            user_phrase = st_keyup(
                "Type at least two words:",
                value="",
                placeholder="e.g. the automaton",
                key="user_phrase_input"
            )
        with col2:
            top_n = st.number_input(
                "Max suggestions to show:",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )

        # Autocomplete suggestions appear automatically while typing.
        tokens = tokenize(user_phrase)
        if len(tokens) < 2:
            st.info("Enter at least two words to get suggestions.")
        else:
            suggestions, current_state, message = get_next_word_suggestions(
                pfa=st.session_state.pfa,
                user_input=user_phrase,
                top_n=top_n
            )

            if current_state in st.session_state.pfa and suggestions:
                # Show subtle state info line above quick suggestions
                st.caption(f"Suggestions for state: ({current_state[0]}, {current_state[1]})")

                # Show suggested next words in one horizontal line as chips
                chips_html = '<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 5px; margin-bottom: 15px; align-items: center;">'
                for idx, (word, prob) in enumerate(suggestions):
                    if idx == 0:
                        # Highlight the highest probability suggestion visually
                        chips_html += f'<div style="background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 6px 14px; border-radius: 16px; font-weight: bold; border: 1px solid #1d4ed8; box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3); font-family: sans-serif; font-size: 0.95em;">{word}</div>'
                    else:
                        # Normal suggestions
                        chips_html += f'<div style="background-color: rgba(128, 128, 128, 0.1); color: inherit; padding: 6px 14px; border-radius: 16px; border: 1px solid rgba(128, 128, 128, 0.2); font-family: sans-serif; font-size: 0.95em;">{word}</div>'
                chips_html += '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)

                # Keep sentence preview
                top_word = suggestions[0][0]
                cleaned_input = user_phrase.strip()
                st.markdown(f"📝 **Sentence Preview:** *\"{cleaned_input} **{top_word}**\"*")

                # Keep detailed table, but rename heading
                st.markdown("### Detailed Probability Table")

                table_data = []
                for idx, (word, prob) in enumerate(suggestions):
                    table_data.append({
                        "Rank": idx + 1,
                        "Suggested Next Word": word,
                        "Probability": f"{prob:.4f} ({prob:.2%})"
                    })

                df_suggestions = pd.DataFrame(table_data)
                st.table(df_suggestions)

                # Equal probability warning
                probs = [prob for word, prob in suggestions]
                if len(probs) > 1 and len(probs) != len(set(probs)):
                    st.caption("ℹ️ *Note: Multiple suggestions have the same probability, indicating equal likelihood for those transitions.*")

    st.markdown("---")
    st.markdown("##### 📖 How Suggestions Work")
    st.markdown(
        """
        * Predictions are generated using learned transition probabilities.
        * The next word depends only on the current bigram state (Markov property).
        * Unknown word combinations cannot produce suggestions because they were not seen during training.
        """
    )

# Footer displayed on all pages
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey; font-size:0.85em;'>"
    "🎓 <strong>Theory of Computation</strong> &nbsp;|&nbsp; "
    "Probabilistic Finite Automata Based Text Generator &nbsp;|&nbsp; "
    "Bigram Markov Chain Model"
    "</div>",
    unsafe_allow_html=True,
)

