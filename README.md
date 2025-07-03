# Host-Phage Interaction Prediction: Reproduction of Research Results

This repository reproduces the main findings from [Gaborieau et al. (2024)](https://www.nature.com/articles/s41564-024-01832-5) on host-phage interaction prediction.
The implementation includes data loading, preprocessing, feature engineering, model training, evaluation, and result visualization.

A comprehensive summary of the code and findings is available [here](https://theafox.vercel.app/mlaa).

This repository is a fork of the original research paper's codebase.
The main experiments are implemented in the `experiments.ipynb` [Jupyter notebook](https://jupyter.org/).
The code in the `dev/` and `data/` directories remains largely unchanged from the original and serves as a reference for the implementation.
For detailed information about modifications, please consult the repository's commit history.

> [!NOTE]
>
> For anyone interested in working with the `fasta` files, you'll need to concatenate and extract them like so:
>
> ```sh
> cat data/fasta_files/*zip_part* > data/fasta_files/fasta.zip
> unzip data/fasta_files/fasta.zip -d data/fasta_files/
> ```
