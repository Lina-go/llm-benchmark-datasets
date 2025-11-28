"""
Prompt template for single-label (multinomial) classification tasks.
"""
from typing import List, Optional
from prompts.base_prompt import BasePromptTemplate


class MultinomialPromptTemplate(BasePromptTemplate):
    """
    Prompt template for single-label classification tasks.
    
    This template creates prompts that ask the LLM to classify text
    into exactly ONE category from a predefined set.
    """
    
    def __init__(self, available_labels: List[str], language: str = "es"):
        """
        Initialize the single-label prompt template.
        
        Args:
            available_labels: List of possible labels
            language: Prompt language ('es' or 'en')
        """
        super().__init__(language=language)
        self.available_labels = available_labels
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for single-label classification.
        
        Args:
            text: Text to classify
            **kwargs: Additional parameters (unused)
            
        Returns:
            Formatted prompt as string
        """
        labels_str = "\n".join(f"- {label}" for label in self.available_labels)
        
        if self.language == "es":
            prompt = f"""Eres un clasificador de textos académicos. Tu tarea es clasificar el siguiente texto en exactamente UNA de las categorías disponibles.

Categorías disponibles:
{labels_str}

Texto a clasificar:
{text}

Instrucciones:
- Responde ÚNICAMENTE con el nombre exacto de la categoría
- Debes elegir exactamente UNA categoría
- No agregues explicaciones ni texto adicional

Categoría:"""
        else:
            prompt = f"""You are an academic text classifier. Your task is to classify the following text into exactly ONE of the available categories.

Available categories:
{labels_str}

Text to classify:
{text}

Instructions:
- Respond ONLY with the exact category name
- You must choose exactly ONE category
- Do not add explanations or additional text

Category:"""
        
        return prompt
    
    def parse_response(self, response: str) -> str:
        """
        Parse the model's response to extract predicted label.
        
        Args:
            response: Raw model response
            
        Returns:
            Predicted label as string
        """
        response = response.strip().lower()
        
        # Try to match with available labels
        for label in self.available_labels:
            if label.lower() == response:
                return label
            if label.lower() in response:
                return label
        
        # If no match, return cleaned response
        return response
    
    def __repr__(self) -> str:
        return (
            f"MultinomialPromptTemplate(\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  language='{self.language}'\n"
            f")"
        )