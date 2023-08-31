# article-gyre-carbon-diagnostics
Python scripts used to plot the figure of the article on the sensitivity to resolution of the ocean carbon uptake.

- main_article_gyre_carbon_cc_v4.py
  > produce figure 1 to 4
- main_carbon_box_budget.py
  > compute the carbon budget for producing figure 4
- main_carbon_box_budget_cc27_dictrd_correction_missing_values.py
  > correct some lost data for the simulation climate change at 1/27° resolution. Year 66, 67 and 68 of the simulation are linearly interpolated.
- FGYRE
  > contains python functions used in the scripts
