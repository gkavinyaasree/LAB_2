# app.py
import random
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters (updated as requested) ----
POP_SIZE = 300          # 1) population setup
CHROM_LEN = 50          # 3) length of individuals
TARGET_ONES = 80        # 2) optimum at ones == 80
MAX_FITNESS = 80        # 4) return value is 80 when ones == 50
N_GENERATIONS = 50      # 5) number of generations

# ---- GA Hyperparameters (simple & sensible) ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN  # ~1 bit flip expected per chromosome
ELITISM = 2                      # best 2 individuals copied unchanged

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """Peak value MAX_FITNESS (80) when number of ones == TARGET_ONES (50)
        Each 1 away from 50 (more OR fewer) costs 1 point."""
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# ---- GA Operators ----
def init_population() -> np.ndarray:
    """Random 0/1 matrix of shape (300, 80)."""
    return np.random.randint(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

def tournament_selection(pop, fits):
    """Pick K random individuals, return a copy of the fittest."""
    idx = np.random.randint(0, len(pop), size=TOURNAMENT_K)
    return pop[idx[np.argmax(fits[idx])]].copy()

def single_point_crossover(p1, p2):
    """Swap tails of two parents after a random cut point."""
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)  # cut between [1 .. CHROM_LEN-1]
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(ind):
    """Flip each bit with probability MUTATION_RATE."""
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    ind[mask] = 1 - ind[mask]
    return ind


# ---- Main Evolution Loop ----
def evolve(pop):
    history = {"generation": [], "best": [], "average": [], "worst": [], "best_ones": []}
    best_ind, best_f = None, -np.inf

    for gen in range(1, N_GENERATIONS + 1):
        # (a) Evaluate fitness
        fits = np.array([fitness(ind) for ind in pop])

        # (b) Record statistics
        i = int(np.argmax(fits))
        history["generation"].append(gen)
        history["best"].append(float(fits[i]))
        history["average"].append(float(fits.mean()))
        history["worst"].append(float(fits.min()))
        history["best_ones"].append(int(pop[i].sum()))
        if fits[i] > best_f:
            best_f, best_ind = float(fits[i]), pop[i].copy()

        # (c) Elitism: carry over the best individuals unchanged
        elite_idx = np.argsort(fits)[-ELITISM:] 
        new_pop = [pop[i].copy() for j in elite_idx]

        # (d) Selection -> Crossover -> Mutation until population is full
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = single_point_crossover(p1, p2)
            new_pop.extend([mutate(c1), mutate(c2)])    

        pop = np.array(new_pop[:POP_SIZE], dtype=np.int8)

    return best_ind, best_f, pd.DataFrame(history)

# ---- Streamlit User Interface ----
st.set_page_config(page_title="Bit Pattern GA (80 bits, target ones = 50)", page_icon="🧬", layout="centered")
st.title("🧬 Genetic Algorithm: Evolve an 80-bit Pattern")
st.caption("BSD3513 Lab Report 2 - Chapter 2: Search Algorithms")

with st.expander("ℹ️ Problem setup (fixed by the question)", expanded=True):
    st.markdown(
        f"""
- **Population size**: `{POP_SIZE}`  
- **Chromosome length**: `{CHROM_LEN}` bits 
- **Optimum**: number of ones = `{TARGET_ONES}`, giving fitness = {MAX_FITNESS} 
- **Fitness function**: `{MAX_FITNESS}` (when ones = {TARGET_ONES})  
- **Generations**: `{N_GENERATIONS}`  
- **Selection**: Tournament (k={TOURNAMENT_K})  
- **Crossover**: Single-point (rate={CROSSOVER_RATE})  
- **Mutation**: Bit-flip (rate={MUTATION_RATE:.4f} per bit)
- **Elitism**: Top `{ELITISM}` individuals preserved
"""
    )

seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)

if st.button("Run Genetic Algorithm", type="primary"):
    # Reproducibility
    random.seed(int(seed))
    np.random.seed(int(seed))

    # Initialize and evolve
    best_ind, best_fit, history = evolve(init_population())
    ones = int(best_ind.sum())

    # Results
    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Best Individual Found")
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Best Fitness", value=f"{best_fit:.0f} / {MAX_FITNESS}")
    c2.metric(label="Ones", value=ones_count)
    c3.metric(label="Zeros", value=zeros_count)
    st.code("".join(map(str, best_ind.tolist())), language="text")

    # Convergence plot
    st.subheader("📉 Convergence (Best Fitness per Generation)")
    fig, ax = plt.subplots()
    ax.plot(history["generation"], history["best"], label="Best", linewidth=2)
    ax.plot(history["generation"], history["average"], label="Average", linestyle="--")
    ax.plot(history["generation"], history["worst"], label="Worst", linestyle=":")
    ax.axhline(y=MAX_FITNESS, color="green", alpha=0.4, label="Max possible (80)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Convergence")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    st.subheader("📉 Generation-by-generation table")
    st.dataframe(history, use_container_width=True, hide_index=True)

    # Validation note
    if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
        first = int(history.loc[history["best"] == MAX_FITNESS, "generation"].iloc[0])
        st.success(f"Perfect solution: ones = {TARGET_ONES}, fitness = {MAX_FITNESS} ✅"
                   
                   f" (first found at generation {first}).")
    else:
        st.info("GA may reach near-optimal solutions; try another seed to explore further.")

st.caption("© 2025 Simple GA demo (80 bits, optimum at 50 ones).")