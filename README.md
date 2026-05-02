# Lab 3 - ANOVA Analysis

This repository contains the source code for Lab 3 of KQC7016 Data Analytics.

## Project Description

This Lab 3 project uses the WorldEnergy dataset to conduct a one-way ANOVA analysis. The analysis compares the mean renewable electricity share among selected Southeast Asian countries from 1985 to 2024.

## Research Question

Is there a statistically significant difference in mean renewable electricity share among the selected Southeast Asian countries?

## Files

- `lab3_anova.py`: Main Python script for ANOVA analysis.
- `WorldEnergy.csv`: Dataset used in this lab.
- `outputs/`: Folder containing generated tables and figures.
- `requirements.txt`: Required Python libraries.

## Main Functions

- `load_dataset()`: Loads the WorldEnergy dataset.
- `preprocess_data()`: Filters selected countries and years, and prepares data for ANOVA.
- `descriptive_statistics()`: Generates descriptive statistics for each country.
- `run_anova()`: Runs one-way ANOVA and calculates effect size.
- `create_visualizations()`: Creates boxplot, trend chart, and mean comparison chart.
- `save_outputs()`: Saves cleaned data, descriptive statistics, and ANOVA table.
- `print_final_guide()`: Prints a short guide for report discussion.

## How to Run

```bash
python lab3_anova.py
