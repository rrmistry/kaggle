# Create Anaconda Environment
conda create -n tensorflow pip python=3.5

# Switch to newly created environment
activate tensorflow

# Update pip version to latest
python -m pip install --upgrade pip

# Install pandas library
conda install pandas

# Install jupyter notebook
conda install jupyter

# Install tensorflow
conda install tensorflow
conda install tensorflow-gpu

# Update python linter for better intellisense
conda install pylint

