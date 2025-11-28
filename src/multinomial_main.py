"""
Main runner for single-label classification with Weave tracking.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse

import weave

from evaluation.multinomial_predictor import MultinomialPredictor
from models.huggingface_llm import HuggingFaceLLM
from models.llm_multinomial_model import LLMMultinomialModel
from prompts.multinomial_prompt import MultinomialPromptTemplate
from utils.multinomial_datareader import AbstractRosarioDataset


# Weave project name
WEAVE_PROJECT = "scibeto-benchmark-evaluation"


def main():
    parser = argparse.ArgumentParser(description="Run single-label classification")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name")
    parser.add_argument("--data-dir", type=str, default="data/abstracts_rosario",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/multinomial",
                        help="Path to save results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for testing")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "parquet"])
    
    args = parser.parse_args()
    
    # Initialize Weave
    weave.init(WEAVE_PROJECT)
    
    print("=" * 60)
    print("SINGLE-LABEL CLASSIFICATION - ABSTRACT ROSARIO")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = AbstractRosarioDataset(data_dir=args.data_dir)
    print(dataset)
    
    # Create LLM
    print("\n[2/4] Loading LLM...")
    llm = HuggingFaceLLM(
        model_name=args.model,
        load_in_4bit=True
    )
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    prompt_template = MultinomialPromptTemplate(
        available_labels=dataset.labels,
        language="es"
    )
    
    # Create model
    model = LLMMultinomialModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template
    )
    print(model)
    
    # Create predictor and save results
    print("\n[4/4] Generating predictions...")
    predictor = MultinomialPredictor(model, dataset)
    
    output_subdir = f"{args.output_dir}/{args.model.replace('/', '_')}"
    predictor.save_predictions(
        split=args.split,
        output_dir=output_subdir,
        max_samples=args.max_samples,
        format=args.format
    )
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()