dolos_ml_scripts/  
    - config.py  
    - io_utils.py  
    - preprocess_dolos.py  
    - agg_static.py  
    - agg_temporal.py  
    - pca_step.py  
    - model_logistic.py  
    - model_stacking_tree.py  
    - model_lasso.py  
    - evaluate.py  
    - run_experiments.py  




| experiment               | acc     | auc     |
|-------------------------|---------|---------|
| temporal + logistic      | 0.615385 | 0.670259 |
| temporal + lasso         | 0.597285 | 0.665246 |
| both + logistic          | 0.642534 | 0.648120 |
| both + lasso             | 0.628959 | 0.644612 |
| both + pca_logistic      | 0.552036 | 0.607268 |
| static + logistic        | 0.556561 | 0.581621 |
| static + lasso           | 0.565611 | 0.581203 |
| static + pca_logistic    | 0.520362 | 0.567502 |
| both + stacking_tree     | 0.515837 | 0.557811 |
| temporal + stacking_tree | 0.452489 | 0.554052 |
| temporal + pca_logistic  | 0.561086 | 0.549039 |
| static + stacking_tree   | 0.484163 | 0.536926 |
