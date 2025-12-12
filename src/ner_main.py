"""
Main runner for NER with Weave tracking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
from dotenv import load_dotenv
import random
from typing import List, Dict
load_dotenv()

import weave

from evaluation.ner_predictor import NERPredictor
from models.llm_ner_model import LLMNERModel
from prompts.ner_prompt import NERPromptTemplate
from utils.ner_datareader import EconIEDataset

WEAVE_PROJECT = "scibeto-benchmark-evaluation"


def get_llm(model_name: str, provider: str):
    if provider == "openai":
        from models.openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_name)
    elif provider == "huggingface":
        from models.huggingface_llm import HuggingFaceLLM
        if "GPTQ" in model_name or "gptq" in model_name:
            return HuggingFaceLLM(model_name=model_name, load_in_4bit=False)
        return HuggingFaceLLM(model_name=model_name, load_in_4bit=True)
    else:
        raise ValueError(f"Provider '{provider}' not supported")


def get_few_shot_examples(
    dataset: EconIEDataset, 
    n_examples: int
) -> List[Dict]:
    """
    Get few-shot examples from train set.
    
    Returns examples in format: [{"text": ..., "entities": [...]}, ...]
    """
    if dataset.train is None:
        raise ValueError("Train split required for few-shot examples")
    
    sentences = dataset.get_text_sentences("train")
    tokens_list, tags_list = dataset.get_sentences_and_labels("train")
    
    # Convertir a formato de entidades
    examples = []
    for sent, tokens, tags in zip(sentences, tokens_list, tags_list):
        entities = bio_to_entities(tokens, tags)
        # Solo incluir ejemplos con entidades
        if entities:
            examples.append({
                "text": sent,
                "entities": entities
            })
    
    # Seleccionar ejemplos aleatorios
    random.seed(42)
    if len(examples) > n_examples:
        examples = random.sample(examples, n_examples)
    
    return examples[:n_examples]


def bio_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    """Convierte tags BIO a lista de entidades."""
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_entity
                })
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_entity
                })
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append({
            "text": " ".join(current_tokens),
            "type": current_entity
        })
    
    return entities


def main():
    parser = argparse.ArgumentParser(description="Run NER extraction")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "huggingface"],
                        help="Model provider")
    parser.add_argument("--data-dir", type=str, default="data/ner_econ_ie",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/ner",
                        help="Path to save results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for testing")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Number of few-shot examples (0 = zero-shot)")
    
    args = parser.parse_args()
    
    weave.init(WEAVE_PROJECT)
    
    print("=" * 60)
    print("NER - ECON-IE")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = EconIEDataset(data_dir=args.data_dir)
    print(dataset)
    
    # Create LLM
    print(f"\n[2/4] Loading LLM ({args.provider})...")
    llm = get_llm(args.model, args.provider)
    print(llm)
    
    # Determine shot type
    shot_type = f"few_shot_{args.few_shot}" if args.few_shot > 0 else "zero_shot"
    
    # Get few-shot examples if needed
    few_shot_examples = None
    if args.few_shot > 0:
        print(f"\n[2.5/4] Getting {args.few_shot} few-shot examples...")
        few_shot_examples = get_few_shot_examples(dataset, args.few_shot)
        print(f"  Got {len(few_shot_examples)} examples")
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    prompt_template = NERPromptTemplate(
        entity_types=dataset.entity_types,
        language="es",
        examples=few_shot_examples
    )
    print(prompt_template)
    
    # Create model
    model = LLMNERModel(
        llm=llm,
        entity_types=dataset.entity_types,
        prompt_template=prompt_template
    )
    print(model)
    
    # Create predictor and save results
    print("\n[4/4] Generating predictions...")
    predictor = NERPredictor(model, dataset)
    
    output_subdir = f"{args.output_dir}/{args.model.replace('/', '_')}/{shot_type}"
    predictor.save_predictions(
        split=args.split,
        output_dir=output_subdir,
        max_samples=args.max_samples,
        shot_type=shot_type
    )
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()