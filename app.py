import json
import numpy as np
import torch
from transformers import pipeline, AutoModelForSequenceClassification

class InferlessPythonModel:
    def initialize(self):
        try:
            self.generator = AutoModelForSequenceClassification.from_pretrained('jinaai/jina-reranker-v1-turbo-en', num_labels=1, trust_remote_code=True) 
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            self.generator = None
    
    def infer(self, input_data):
        if isinstance(input_data, dict):
            input_json = input_data
        else:
            input_json = json.loads(input_data)
        query = input_json['query']
        chunks = input_json['chunks']
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("Chunks must be a list of strings")
        combined_input = [[query, chunk] for chunk in chunks]
        
        results = self.generator.compute_score(combined_input)
        
        print("Results structure:", results, flush=True)

        # Make key-value pairs for the score
        key_val = []
        for i in range(len(results)):
            key_val.append({"chunk": chucks[i], "score" : results[i] })

        # Sort according to score        
        try:
            ranked_results = ranked_results = sorted(key_val, key=lambda x: x['score'], reverse=False)
        except (TypeError, KeyError, IndexError) as e:
            print(f"Error during sorting: {e}", flush=True)
            ranked_results = key_val
            
        result_texts = []
        result_scores = []

        for result in ranked_results:
                result_texts.append(result['chunk'])
                result_scores.append(result['score'])
            else:
                print(f"Unexpected result format: {result}", flush=True)

        return {"result": result_texts, "scores" results}


    def finalize(self):
        self.generator = None
        print("Pipeline finalized.", flush=True)
