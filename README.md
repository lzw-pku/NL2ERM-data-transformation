# NL2ERM-data-transformation

The data transformation scripts to transform datasets of NL2SQL into datasets of NL2ERM.

## NL2SQL Dataset

Download the Spider dataset from [here](https://yale-lily.github.io/spider) and unzip it to ./data directory.

The code in preprocess.py is for Spider dataset.
You can also use other NL2SQL datasets, but you need to modify the code in preprocess.py.

## Data Transformation

1. python3 preprocess.py
2. python3 erd_generate.py

The generated NL2ERM dataset is in ./data/ directory.