#!/bin/sh

# to execute the code use: ./run.sh
# You have to export the AWS bedrock credentials (for bedrock models) or your desired API keys before running the code.

## to run on full set
# python3 faithfulness_evaluation.py --model=zero-shot --dataset_name=AggreFact-CNN --output_file_path=test.json 

## to run on sample data
python3 faithfulness_evaluation.py --model=zero-shot --data_file=sample_data.json --output_file_path=test.json --benchmark
