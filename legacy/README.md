# Legacy: Module 5 analysis scripts (model v1.x)

These are the scripts that produced the Module 5 presentation figures. They are kept unchanged for
traceability. They are superseded by the `careconnect/` package (model v2.0.0), which:

* rebuilds the Module 3 pipeline and Module 4 training code that were not preserved in any repository;
* corrects the prior-attendance features (v1 counted earlier-*scheduled* appointments whose outcome
  was not yet known at booking time - 29,727 of 110,514 rows affected);
* tunes on the full training set with 5-fold CV (v1 used a 25,000-row sample with 3-fold CV);
* logs everything to MLflow and adds LIME, counterfactuals, monitoring and a promotion gate.

`build_deck.js` rebuilds the Module 5 deck from the v1 artifacts.
