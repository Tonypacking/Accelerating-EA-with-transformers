"""Minimal end-to-end check of the COCO toolchain.

Runs random search with a tiny budget on a few bbob problems, then
post-processes the results with cocopp. Not a real experiment - only
verifies that cocoex -> data -> cocopp works.
"""

import cocoex
import cocopp
import numpy as np

BUDGET_MULTIPLIER = 5  # evaluations per dimension, tiny on purpose

suite = cocoex.Suite("bbob", "", "function_indices:1,2 dimensions:2,3 instance_indices:1")
observer = cocoex.Observer("bbob", "result_folder: smoke_test algorithm_name: random_search")

rng = np.random.default_rng(42)
for problem in suite:
    problem.observe_with(observer)
    for _ in range(BUDGET_MULTIPLIER * problem.dimension):
        x = rng.uniform(problem.lower_bounds, problem.upper_bounds)
        problem(x)

print(f"\nExperiment data written to: {observer.result_folder}")
cocopp.genericsettings.interactive_mode = False  # don't auto-open the HTML report
cocopp.main(observer.result_folder)
print("\nSmoke test finished - post-processing output in ppdata/")
