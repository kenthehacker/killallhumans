# A Method to Speed Up Convergence of Iterative Learning Control for High Precision Repetitive Motions

- **URL**: https://arxiv.org/abs/2307.15912
- **Authors**: Richard W. Longman, Shuo Liu, Tarek A. Elsharhawy
- **Year**: 2023
- **Venue**: arXiv (eess.SY)

## Key Contribution
This paper proposes using a model-based warmstarting strategy to accelerate ILC convergence for high-precision repetitive motions. Instead of running all ILC iterations on hardware (or in a full simulation), the authors suggest first running multiple iterations against a mathematical model of the system. The model-converged offset profile is then used as the initial feedforward for hardware/sim iterations, dramatically reducing the number of expensive real-world iterations needed. This is conceptually similar to transfer learning: learn quickly in a cheap environment, then fine-tune in the real environment.

The paper also provides criteria for when to switch from model-based to hardware iterations — specifically, when the model-based error stops decreasing (model error floor), that's the optimal transfer point. Further model iterations would overfit to model inaccuracies.

## Technical Approach
The approach builds on standard ILC update: u_{j+1}(k) = Q[u_j(k) + L*e_j(k)], where Q is a robustness filter and L is the learning gain. The key insight is that running ILC on a model first eliminates the "easy" error (systematic model-predictable error) in O(10) fast numerical iterations, leaving only the model-mismatch residual for hardware iterations. The model-based iterations are computationally trivial compared to hardware experiments.

The transfer criterion is: stop model-based iterations when ||e_{j+1}||/||e_j|| > (1 - ε) for some small ε, indicating the learning is stalling on model error. Then deploy the learned feedforward to hardware and continue ILC there from this warm start.

## Results
The paper demonstrates on spacecraft scanning maneuvers that model-based warmstarting reduces hardware iterations by 60-80% while achieving the same final tracking precision. The key finding is that model-based learning converges in ~5-10 iterations (fast, cheap), and the transferred profile brings hardware error to within 2-3x of the final achievable precision, requiring only 2-3 hardware iterations to converge.

## Relevance to Our System
Highly relevant. Our ILC runs 5 iterations in an inner simulation (with kp=6, kd=5) and then applies the result to the benchmark simulation (kp=7, kd=5.5). This is already a form of model-based warmstarting, but the inner sim parameters don't match the outer sim. The "beneficial mismatch" we discovered in iteration 39 (where synchronizing gains regressed performance) is consistent with Longman's finding that model-based ILC converges to a different error floor than hardware ILC.

The paper suggests we could potentially improve convergence by running more model-based iterations (e.g., 8-10) before applying to the benchmark sim, since we currently cap at 5. However, iteration 35 showed that 8 iterations caused cumulative offset saturation. The key is that Longman advocates for a stopping criterion based on diminishing returns, not a fixed iteration count.

## Actionable Takeaways
1. Consider adding a convergence-based stopping criterion to our ILC instead of fixed 5 iterations — stop when error reduction per iteration drops below a threshold.
2. The model-mismatch between ILC inner sim (kp=6) and benchmark (kp=7) is a feature not a bug — consistent with Longman's framework where model-based pre-convergence handles systematic error.
3. Could increase ILC iterations from 5 to 7-8 IF we add a per-section convergence check that stops sections individually when they converge, avoiding the cumulative offset saturation problem from iter 35.
4. The transfer criterion concept could be applied per-section: let each section run until convergence rather than using a global iteration count.

## Limitations & Caveats
- The paper assumes a linear or nearly-linear system, which our PD-controlled drone approximates well.
- The spacecraft scanning application has much slower dynamics than our racing drone.
- We already use 5 iterations which is in the range the paper suggests for model convergence.

## Key Parameters / Constants
- Typical model convergence: 5-10 iterations
- Typical hardware-needed iterations after warmstart: 2-3
- Transfer criterion threshold ε: 0.01-0.05 (problem-dependent)
- Total iteration savings: 60-80%
