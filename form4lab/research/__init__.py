"""Anti-overfitting research harness: an example train/validate/(locked-)test
loop over form4lab.scoring.portfolio_simulator, scored with the Deflated
Sharpe Ratio (Bailey & Lopez de Prado, 2014) to penalize repeated trials.

Ships as a skeleton: form4lab.research.space has no atoms, signals, or banned
regions beyond six textbook literature entries and the one shipped example
strategy. Point the FORM4LAB_RESEARCH_SPACE env var at your own module (same
shape as form4lab.research.space) once you have findings of your own to
encode — see form4lab.research.loop's module docstring for the workflow.
"""
