# On the Choice of Filtfilt, Circulant, and Cliff Filters for Robustification of ILC

- **URL**: https://www.researchgate.net/publication/338198028_On_the_Choice_of_Filtfilt_Circulant_and_Cliff_Filters_for_Robustification_of_Iterative_Learning_Control
- **Authors**: Richard W. Longman (Columbia University) and colleagues
- **Year**: 2019
- **Venue**: Advances in the Astronautical Sciences (AAS/AIAA Space Flight Mechanics Meeting), Vol. ~170 (December 2019)

---

## Key Contribution

The paper provides a rigorous comparison of three zero-phase low-pass Q-filters used to robustify Iterative Learning Control (ILC) against high-frequency model uncertainty: the **filtfilt** filter (MATLAB's standard zero-phase filtering command), the **circulant filter** (a steady-state Toeplitz-to-circulant approximation), and the **cliff filter** (an ideal sharp-cutoff spectral filter). The central result is a proof that the cliff filter is a special case of the circulant filter family, and a numerical evaluation suggesting the cliff filter produces the best RMS tracking performance among the three. The paper also identifies a concrete failure mode of `filtfilt`—namely, that MATLAB's default initial condition choice can cause ILC instability—which motivates the alternatives.

The paper is foundational for anyone implementing Q-filter-based ILC robustification because it clarifies what each filter actually does to a finite-length signal (as opposed to an infinite periodic signal assumed in classical frequency-domain analysis), exposing pitfalls that standard textbook treatments do not address.

---

## Technical Approach

### Why Q-filters are needed in ILC

Standard ILC uses a learning update of the form:

```
u_{k+1} = Q(u_k + L * e_k)
```

where `u_k` is the input on trial `k`, `e_k` is the tracking error, `L` is the learning gain, and `Q` is the Q-filter (a low-pass robustification filter). Without `Q`, ILC attempts to drive error to zero at all frequencies up to Nyquist. However, high-frequency parasitic modes, unmodeled resonances, or discretization artifacts make the high-frequency inverse model `L` unreliable. The Q-filter attenuates these unstable high-frequency corrections, trading off convergence to zero error above the cutoff for stability.

The convergence condition for this ILC update law is:

```
||Q(I - LG)||_∞ < 1
```

where `G` is the plant transfer matrix (here treated as a matrix of Markov parameters), `L` is designed to make `I - LG` small in the passband, and `Q` ensures the product has magnitude below 1 everywhere. In the passband (below cutoff), `Q ≈ 1`, so `I - LG` must be small (good model matching). In the stopband (above cutoff), `Q ≈ 0`, so model error above cutoff does not destabilize the system.

### The finite-time ILC problem structure

For a system with `N` time steps, the full-trial ILC problem is formulated in terms of the **Toeplitz matrix** `H` of Markov parameters (impulse response coefficients). The Toeplitz structure means that classical DFT frequency analysis does not apply exactly—the matrix is not circulant, so the DFT eigenvectors are not exact eigenvectors of `H`. This is the root cause of discrepancy between what the three filters do in theory versus in practice on a finite-length trial.

### Filtfilt filter

`filtfilt` applies a causal IIR filter (e.g., Butterworth) in the forward direction, then applies it again in the reverse direction. This yields zero-phase response because the phase shifts cancel. However, because the signal has finite length, the filter must be initialized at the signal boundaries, and MATLAB's default initialization (based on steady-state of a step input) introduces **transient distortions** at the start and end of each filtered signal. The paper notes a specific historical case where this initialization caused ILC **instability**—the transient pushed the effective frequency response outside the stability region for certain ILC designs.

Butterworth is the recommended causal filter to feed into filtfilt because:
- It maintains gain at or below unity throughout the passband (no Chebyshev ripple that could exceed 1)
- This allows a higher cutoff frequency before the convergence condition is violated
- The design pipeline is: choose analog Butterworth → discretize via Tustin transformation with frequency pre-warping → apply filtfilt

### Circulant filter

The circulant filter starts from the same Butterworth design but constructs the associated **circulant matrix** from the Toeplitz matrix of Markov parameters. The circulant approximation is valid when the number of time steps `N` is large enough that the circular wrap-around does not significantly corrupt the finite-time response (specifically, when `N` is at least several times the system settling time).

The key property of a circulant matrix is that it is exactly diagonalized by the DFT—its eigenvalues are precisely the DFT of its first row, which are the steady-state frequency response samples at the `N` observable frequencies `f_k = k/(N * dt)` for `k = 0, 1, ..., N-1`. Therefore:
- The circulant filter gives exactly the steady-state Butterworth response at each of these frequencies
- It eliminates the initial-condition transient issues of `filtfilt`
- It is computed as: take DFT of signal, multiply by the DFT of the filter impulse response (evaluated at the `N` frequencies), take inverse DFT

This is equivalent to circular (periodic) convolution. The Gibbs phenomenon can appear if the signal's start and end values are not equal, since circular convolution implicitly assumes periodicity.

### Cliff filter

The cliff filter uses an **ideal rectangular spectral window**: gain of exactly 1 below the cutoff frequency and exactly 0 above it, with zero phase at all frequencies. This is implemented as:
1. Compute DFT of the signal
2. Zero out all frequency components above the cutoff (index `k_c = floor(f_c * N * dt)`)
3. Take inverse DFT

The cliff filter is proven to be a special case of the circulant filter with the ideal rectangular frequency response. Its advantages are:
- Maximally sharp cutoff—allows the highest possible learning bandwidth while completely suppressing above-cutoff content
- No Butterworth approximation error in the passband (gain is exactly 1, not approximately 1)
- Zero Gibbs ringing within the passband (the ringing is in the stopband where the filter response is zero, so it doesn't affect the learned correction)

The paper's main theoretical result is the proof that the cliff filter is a circulant filter—specifically, the circulant filter whose frequency profile is the ideal rectangular function sampled at the `N` DFT frequencies.

### Numerical comparison

The paper includes a table of RMS errors comparing the three filter outputs against the desired steady-state output (the Butterworth filtered reference). Results show:
- The cliff filter achieves the lowest RMS tracking error
- The circulant filter (with Butterworth profile) is close but slightly worse due to Butterworth transition-band rolloff
- `filtfilt` performs worst, primarily due to boundary transient effects

A secondary comparison examines convergence speed across ILC iterations. The cliff filter's sharper cutoff allows it to retain more correction content in the passband, which translates to faster convergence when the cutoff is near the region of model uncertainty.

---

## Results

- The cliff filter achieves strictly lower RMS error than both filtfilt and the circulant Butterworth filter in numerical experiments
- Proof establishes the cliff filter as an ideal-profile circulant filter, unifying the three filter types in a common framework
- `filtfilt` is shown to have an initialization-dependent failure mode that can destabilize ILC; this failure was confirmed in real hardware experiments referenced in earlier Longman group work
- The circulant filter eliminates filtfilt's boundary issues but inherits Butterworth's transition band
- Both filtfilt and circulant still exhibit Gibbs-like boundary ringing when the signal start and end values are mismatched (which is nearly always the case in ILC iterations)
- Future work is noted: whether the cliff filter permits the highest achievable cutoff in the presence of a given level of high-frequency model error, and its connection to FIR frequency-sampling filter designs in repetitive control

---

## Relevance to Our System

Our current ILC implementation (per-section ILC position-offset table, iteration 26) uses a Q-filter to smooth the learned offsets across gates. The Longman paper is directly applicable:

1. **Filter type selection**: We should prefer the cliff filter (ideal spectral cutoff via DFT zeroing) over a `scipy.signal.filtfilt` implementation. Our gate count (N ~ 20-50 per section) is small enough that boundary transients from filtfilt may meaningfully distort the learned corrections. The cliff filter in DFT space has no such issue.

2. **Cutoff frequency interpretation**: In our spatial ILC, the "frequency" is gates per trial. If we have 20 gates and want to smooth on a scale of 4 gates, the cutoff is at gate-frequency index `k_c = N/4 = 5`, i.e., we zero DFT bins 5 and above. This is a clean implementation in numpy.

3. **Butterworth order for filtfilt baseline**: If we keep a filtfilt implementation for comparison, use a 4th-order Butterworth (order 4 balances rolloff sharpness with stability margin in the convergence condition). The paper implies higher orders push the transition band closer to the stopband but do not necessarily improve convergence if the cliff filter is available.

4. **Convergence condition**: Ensure `||Q(I - LG)||_∞ < 1`. For our diagonal per-gate gain `L = alpha * I`, this simplifies to ensuring `alpha * gain_at_each_frequency < 1/Q_frequency`. Above the cutoff, `Q = 0` so the condition is automatically satisfied. Below cutoff, we need `alpha < 1 / ||G||_passband` which is why conservative step sizes (alpha ~ 0.3-0.5) are standard.

5. **Gibbs phenomenon warning**: Both the circulant and cliff filters assume periodic continuation of the gate error sequence. If our first and last gate errors differ substantially (likely in a non-looping track), we should apply a taper window to the error sequence before DFT-based filtering to suppress Gibbs ringing.

---

## Actionable Takeaways

1. **Replace any filtfilt call in our Q-filter with a DFT-based cliff filter**: In numpy this is trivial—`np.fft.rfft(signal)`, zero indices above `k_c`, `np.fft.irfft(spectrum)`. This is strictly better than `scipy.signal.filtfilt` for finite-length ILC.

2. **Cliff filter cutoff selection rule**: Choose the cutoff gate-frequency `k_c` such that the model error (mismatch between true plant and our trajectory model) first becomes significant. For our min-snap trajectory tracker, we expect model fidelity up to ~3-5 gate frequencies (slow spatial variations are well-modeled; fast gate-to-gate oscillations are not). Set `k_c = 3` to `k_c = 5` depending on track repetitiveness.

3. **Taper the boundary**: Apply a Hann or cosine taper of width 2-3 gates at each end of the error sequence before DFT filtering. This suppresses Gibbs ringing without meaningfully affecting the retained corrections.

4. **Verify convergence condition numerically**: After any gain change, compute `max(|Q_k * (1 - L * G_k)|)` for all DFT frequency bins `k`. If any value exceeds 1, reduce the learning gain `alpha`.

5. **Butterworth order recommendation (if filtfilt is retained for compatibility)**: Use order 4. Higher orders have diminishing returns and can introduce numerical precision issues in the forward-backward filtering at the boundaries.

6. **Do not use filtfilt as the production Q-filter**: The boundary initialization issue identified by Longman is a real concern. Scipy's `filtfilt` uses a different initialization than MATLAB's old implementation but still has end-effect distortions on short sequences.

---

## Limitations & Caveats

1. **Small-N regime**: The circulant approximation is accurate only when `N` significantly exceeds the system settling time. For very short trials (N < ~20), the circulant and cliff filters may behave differently than their infinite-signal analogs. Our gate count per section is in this regime.

2. **Gibbs in cliff filter**: While the cliff filter has no passband Gibbs ringing, the hard spectral cutoff does produce ringing in time near the boundaries. This is why the paper recommends monitoring whether boundary effects contaminate the learned correction.

3. **Linear plant assumption**: All three filters' convergence guarantees are derived for linear time-invariant (LTI) plants. Our drone dynamics are nonlinear (aerodynamic drag, rotor thrust saturation, attitude coupling). The guarantees are approximately valid when linearized around the nominal trajectory but may not hold under large deviations.

4. **No drone-specific experiments**: The paper validates on spacecraft attitude control testbeds. Drone racing involves much faster sampling rates (100-400 Hz vs. ~10 Hz for spacecraft), different noise profiles, and nonminimum-phase dynamics. Cutoff frequency recommendations must be re-derived for our specific system parameters.

5. **Conference paper scope**: This is a conference paper (AAS), not a journal paper. The numerical comparisons are illustrative rather than exhaustive. The claim that cliff filter is strictly best needs to be verified on our specific gate-error sequences.

6. **Future work incomplete at publication**: The paper explicitly states that deeper analysis of the cliff filter's maximum achievable cutoff is deferred to future work. This means we cannot take the "cliff filter is always best" conclusion as unconditional.

---

## Key Parameters / Constants

| Parameter | Value / Recommendation | Source |
|-----------|------------------------|--------|
| Preferred filter type | Cliff filter (ideal DFT spectral cutoff) | Main conclusion |
| Butterworth order (if used) | 4th order | Implied by filtfilt usage section |
| Discretization method | Tustin transform with frequency pre-warping | Filtfilt section |
| Convergence condition | `||Q(I - LG)||_∞ < 1` | Standard ILC theory |
| Cutoff selection rule | Below first frequency where model error is significant | Qualitative guidance |
| Learning gain `alpha` (typical) | 0.3–0.5 (conservative), up to 0.9 (aggressive) | ILC literature norm; not stated explicitly |
| Minimum recommended N for circulant validity | Several times system settling time in samples | Circulant section |
| Gibbs mitigation | Hann taper at signal boundaries | Implied by Gibbs discussion |
| Butterworth passband guarantee | Gain ≤ 1 throughout passband (no ripple) | Stated as reason to prefer Butterworth over Chebyshev |
