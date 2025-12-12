"""
Prompt template for Named Entity Recognition (NER) tasks.
"""
from typing import List, Optional, Dict
from prompts.base_prompt import BasePromptTemplate
import re


class NERPromptTemplate(BasePromptTemplate):
    """
    Prompt template for NER tasks.
    """
    
    def __init__(
        self, 
        entity_types: List[str],
        language: str = "es",
        examples: Optional[List[Dict]] = None
    ):
        """
        Initialize the NER prompt template.
        
        Args:
            entity_types: List of entity types to extract
            language: Prompt language ('es' or 'en')
            examples: Optional few-shot examples [{"text": ..., "entities": [...]}, ...]
        """
        super().__init__(language=language)
        self.entity_types = entity_types
        self.examples = examples or []
    
    @property
    def is_few_shot(self) -> bool:
        return len(self.examples) > 0
    
    def _format_examples(self) -> str:
        """Format few-shot examples."""
        if not self.examples:
            return ""
        
        formatted = []
        for ex in self.examples:
            entities_str = "\n".join(
                f"  - {ent['text']}: {ent['type']}" 
                for ent in ex['entities']
            ) if ex['entities'] else "  (ninguna entidad)"
            
            formatted.append(f"Texto: {ex['text']}\nEntidades:\n{entities_str}")
        
        return "\n\n".join(formatted)
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for NER.
        
        Args:
            text: Text to extract entities from
        """
        types_str = ", ".join(self.entity_types)
        
        few_shot_section = ""
        if self.is_few_shot:
            few_shot_section = f"\n\nEjemplos:\n\n{self._format_examples()}\n\n---"
        
        if self.language == "es":
            prompt = f"""Eres un extractor de entidades nombradas para textos económicos. Tu tarea es identificar y extraer entidades del siguiente texto.

Tipos de entidades a extraer: {types_str}

Instrucciones:
- Extrae TODAS las entidades que encuentres en el texto
- Para cada entidad, indica el texto exacto y su tipo
- Responde en formato JSON: {{"entities": [{{"text": "...", "type": "..."}}]}}
- Si no hay entidades, responde: {{"entities": []}}
- No agregues explicaciones adicionales{few_shot_section}

Texto:
{text}

JSON:"""
        else:
            prompt = f"""You are a named entity extractor for economic texts. Your task is to identify and extract entities from the following text.

Entity types to extract: {types_str}

Instructions:
- Extract ALL entities you find in the text
- For each entity, indicate the exact text and its type
- Respond in JSON format: {{"entities": [{{"text": "...", "type": "..."}}]}}
- If there are no entities, respond: {{"entities": []}}
- Do not add additional explanations{few_shot_section}

Text:
{text}

JSON:"""
        
        return prompt
    
    def parse_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the model's response to extract entities.
        
        Args:
            response: Raw model response
            
        Returns:
            List of entities [{"text": ..., "type": ...}, ...]
        """
        response = response.strip()
        
        # Intentar extraer JSON
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                data = json.loads(json_match.group())
                if "entities" in data:
                    # Validar y limpiar entidades
                    entities = []
                    for ent in data["entities"]:
                        if "text" in ent and "type" in ent:
                            # Normalizar tipo de entidad
                            ent_type = ent["type"].upper()
                            if ent_type in self.entity_types:
                                entities.append({
                                    "text": ent["text"],
                                    "type": ent_type
                                })
                    return entities
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: intentar extraer entidades de texto libre
        return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> List[Dict[str, str]]:
        """Fallback parser for non-JSON responses."""
        entities = []
        
        for entity_type in self.entity_types:
            # Buscar patrones como "Entity: type" o "type: Entity"
            patterns = [
                rf'"([^"]+)":\s*{entity_type}',
                rf'{entity_type}:\s*"([^"]+)"',
                rf'{entity_type}:\s*([^\n,]+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "text": match.strip(),
                        "type": entity_type
                    })
        
        return entities
    
    def __repr__(self) -> str:
        return (
            f"NERPromptTemplate(\n"
            f"  entity_types={self.entity_types},\n"
            f"  language='{self.language}',\n"
            f"  few_shot={self.is_few_shot} ({len(self.examples)} examples)\n"
            f")"
        )