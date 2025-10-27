#!/bin/bash

echo "===== 1. Start Training HerbMind ====="
python3 train.py

echo "===== 2. Start Testing HerbMind ====="
python3 test.py

echo "===== 3. Start Pipeline (Case Studies) ====="
python3 pipeline.py

echo "===== 4. Generate Analysis Figures (t-SNE, Stats) ====="
python3 generate_figures.py

echo "===== All Processes Finished. Check the 'plots' folder. ====="
