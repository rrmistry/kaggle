# Create Anaconda Environment
conda create -n tensorflow pip python=3.5

# Switch to newly created environment
activate tensorflow

# Update pip version to latest
python -m pip install --upgrade pip

# Install pandas library
pip install pandas

# Install jupyter notebook
pip install jupyter

# Update python linter for better intellisense
pip install pylint

# Install Plotting Libraries
pip install matplotlib

# Install tensorflow
pip install --ignore-installed --upgrade tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu

