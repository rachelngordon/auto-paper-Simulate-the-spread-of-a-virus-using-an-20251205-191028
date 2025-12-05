# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_sir(beta, gamma, S0, I0, R0, t_max, dt):
    """Simulate the SIR model using simple Euler integration.

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    S0, I0, R0 : float
        Initial fractions of susceptible, infected, recovered.
    t_max : float
        Maximum time to simulate.
    dt : float
        Time step.

    Returns
    -------
    t : ndarray
        Time points.
    S, I, R : ndarrays
        Fractions of each compartment over time.
    """
    n_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n_steps)
    S = np.empty(n_steps)
    I = np.empty(n_steps)
    R = np.empty(n_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for k in range(1, n_steps):
        dS = -beta * S[k-1] * I[k-1]
        dI = beta * S[k-1] * I[k-1] - gamma * I[k-1]
        dR = gamma * I[k-1]
        S[k] = S[k-1] + dS * dt
        I[k] = I[k-1] + dI * dt
        R[k] = R[k-1] + dR * dt
        # Ensure fractions stay within [0,1]
        S[k] = max(min(S[k], 1.0), 0.0)
        I[k] = max(min(I[k], 1.0), 0.0)
        R[k] = max(min(R[k], 1.0), 0.0)
    return t, S, I, R

def plot_time_series(t, S, I, R, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Fraction of population')
    plt.title('SIR Model Time Series')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_peak_vs_beta(beta_vals, peak_vals, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(beta_vals, peak_vals, marker='o')
    plt.xlabel('Transmission rate (beta)')
    plt.ylabel('Peak infected fraction')
    plt.title('Peak Infected Fraction vs Transmission Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Common parameters
    N = 1.0               # Total population normalized to 1
    I0 = 0.01             # Initial infected fraction
    R0 = 0.0              # Initial recovered fraction
    S0 = N - I0 - R0      # Initial susceptible fraction
    gamma = 0.1           # Recovery rate
    t_max = 160           # Days
    dt = 0.1              # Time step

    # Experiment 1: Baseline SIR dynamics
    beta_baseline = 0.3
    t, S, I, R = simulate_sir(beta_baseline, gamma, S0, I0, R0, t_max, dt)
    plot_time_series(t, S, I, R, "sir_time_series.png")

    # Determine peak infected fraction for baseline (used as final answer)
    peak_infected_baseline = np.max(I)

    # Experiment 2: Transmission rate sweep
    beta_vals = np.arange(0.1, 0.51, 0.02)
    peak_vals = []
    for beta in beta_vals:
        _, _, I_sweep, _ = simulate_sir(beta, gamma, S0, I0, R0, t_max, dt)
        peak_vals.append(np.max(I_sweep))
    peak_vals = np.array(peak_vals)
    plot_peak_vs_beta(beta_vals, peak_vals, "peak_infected_vs_beta.png")

    # Print the primary numeric answer (peak infected fraction for baseline)
    print('Answer:', peak_infected_baseline)

