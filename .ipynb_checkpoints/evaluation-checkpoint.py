import time
import torch
import numpy as np
from tqdm import tqdm

import utils

def evaluate_dataset(model_name, dataset_split, model, tokenizer, dataset, batch_size=16):
    # Evaluation metrics
    correct = 0
    total = len(dataset) * 2 # both clean and corrupted
    predictions = []
    errors_by_count = {}
    errors_by_category = {}
    response_times = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        batch_prompts = []
        batch_true_counts = []
        batch_categories = []
        
        for example in batch:
            # if 'Instruct' in model_name:
            #     clean_formatted_prompt = utils.format_prompt_for_model(example['clean_prompt'], model_name)
            #     corrupted_formatted_prompt = utils.format_prompt_for_model(example['corrupted_prompt'], model_name)
            # else:
            #     clean_formatted_prompt = example['clean_prompt']
            #     corrupted_formatted_prompt = example['corrupted_prompt']
            clean_formatted_prompt = example['clean_prompt']
            corrupted_formatted_prompt = example['corrupted_prompt']
            # clean sample
            batch_prompts.append(clean_formatted_prompt)
            batch_true_counts.append(example['clean_target_count'])
            batch_categories.append(example['category'])
            # corrupted sample
            batch_prompts.append(corrupted_formatted_prompt)
            batch_true_counts.append(example['corrupted_target_count'])
            batch_categories.append(example['category'])
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode responses
        for j, output in enumerate(outputs):
            # Get only the generated part
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract answer
            # if 'Instruct' in model_name:
            #     predicted_count = utils.extract_answer(response)
            # else:
            #     predicted_count = int(response[0])
            predicted_count = int(response[0])
            
            true_count = batch_true_counts[j]
            category = batch_categories[j]
            
            predictions.append({
                'true': true_count,
                'predicted': predicted_count,
                'category': category,
                'response': response
            })
            
            # Update metrics
            if predicted_count == true_count:
                correct += 1
            else:
                # Track errors
                if true_count not in errors_by_count:
                    errors_by_count[true_count] = {'total': 0, 'predictions': []}
                errors_by_count[true_count]['total'] += 1
                errors_by_count[true_count]['predictions'].append(predicted_count)
                
                if category not in errors_by_category:
                    errors_by_category[category] = 0
                errors_by_category[category] += 1
        
        response_times.append(time.time() - start_time)
    
    # Calculate metrics
    accuracy = correct / total
    avg_response_time = np.mean(response_times)
    
    # Analyze error patterns
    error_analysis = utils.analyze_errors(predictions)
    
    results = {
        'model_name': model_name,
        'dataset_split': dataset_split,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_response_time': avg_response_time,
        'errors_by_count': errors_by_count,
        'errors_by_category': errors_by_category,
        'error_analysis': error_analysis,
        'predictions': predictions
    }
    
    return results