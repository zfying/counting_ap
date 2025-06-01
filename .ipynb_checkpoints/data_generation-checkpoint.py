import random
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import os

random.seed(42)

def verify_single_tokens(tokenizer, categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Verify and filter words to ensure they are single tokens."""
    if type(categories) == list:
        single_token_words = []
        for word in categories:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) == 1:
                single_token_words.append(word)
            else:
                print(f"Skipping word: {word} -> {tokens}")
        return single_token_words
    else:
        filtered_categories = {}
        for category, words in categories.items():
            single_token_words = []
            for word in words:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                if len(tokens) == 1:
                    single_token_words.append(word)
                else:
                    print(f"Skipping word: {word} -> {tokens}")
            filtered_categories[category] = single_token_words
        return filtered_categories

def generate_word_list(category_words: List[str], target_count: int, 
                      list_length: int, distractors: List[str]) -> List[str]:
    """Generate a word list with specified target count and total length."""
    if target_count > len(category_words):
        raise ValueError(f"Target count {target_count} exceeds available words {len(category_words)}")
    if target_count > list_length:
        raise ValueError(f"Target count {target_count} exceeds list length {list_length}")
    
    # Select target words
    target_words = random.sample(category_words, target_count)
    
    # Select distractor words
    num_distractors = list_length - target_count
    distractor_words = random.sample(distractors, num_distractors)
    
    # Combine and shuffle
    word_list = target_words + distractor_words
    random.shuffle(word_list)
    
    return word_list

def create_example(category: str, category_words: List[str], target_count: int, 
                  list_length: int, distractors: List[str]) -> Dict:
    """Create a single training example."""
    word_list = generate_word_list(category_words, target_count, list_length, distractors)
    
#     # Create prompt v1
#     prompt = f"""Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
# Type: {category}
# List: [{' '.join(word_list)}]
# Answer: ("""

    # # Create prompt v2
    # prompt = f"Analyze the word list below. Count ONLY words that are {category}.\n" \
    #          f"Follow these rules:\n1. Strictly match the type definition (case-sensitive)\n" \
    #          f"2. Ignore non-word elements\n3. Output ONLY the integer count inside parentheses\n\n" \
    #          f"Type: {category}\nList: [{' '.join(word_list)}]\nAnswer: "

    # Create prompt v3
    prompt = f"Analyze the word list below. Count ONLY words that are {category}.\n" \
             f"Follow these rules:\n1. Strictly match the type definition (case-sensitive)\n" \
             f"2. Ignore non-word elements\n3. Output ONLY one interger count and nothing else\n\n" \
             f"Type: {category}\nList: [{' '.join(word_list)}]\nAnswer: "

    ### Generate corrupted example
    
    # Identify available replacement words
    used_targets = set(word for word in word_list if word in category_words)
    used_distractors = set(word for word in word_list if word not in category_words)
    unused_targets = [w for w in category_words if w not in used_targets]
    unused_distractors = [d for d in distractors if d not in used_distractors]
    
    # Attempt to corrupt a random word
    corrupted_word_list = word_list.copy()
    corrupted_target_count = None
    
    indices = list(range(len(word_list)))
    selected_index = random.choice(indices)
    
    word = corrupted_word_list[selected_index]
    if word in category_words:
        # if random.randint(0, 1) == 1: 
        #     # Replace target with distractor (decreases count)
        #     new_word = random.choice(unused_distractors)
        #     corrupted_word_list[selected_index] = new_word
        #     corrupted_target_count = target_count - 1
        # else:
        #     # Replace target with another target (same count)
        #     new_word = random.choice(unused_targets)
        #     corrupted_word_list[selected_index] = new_word
        #     corrupted_target_count = target_count
        # Replace target with distractor (decreases count)
        new_word = random.choice(unused_distractors)
        corrupted_word_list[selected_index] = new_word
        corrupted_target_count = target_count - 1
    elif word not in category_words:
        # if random.randint(0, 1) == 1: 
        #     # Replace distractor with target (increases count)
        #     new_word = random.choice(unused_targets)
        #     corrupted_word_list[selected_index] = new_word
        #     corrupted_target_count = target_count + 1
        # else:
        #     # Replace distractor with another distractor (same count)
        #     new_word = random.choice(unused_distractors)
        #     corrupted_word_list[selected_index] = new_word
        #     corrupted_target_count = target_count
        # Replace distractor with target (increases count)
        new_word = random.choice(unused_targets)
        corrupted_word_list[selected_index] = new_word
        corrupted_target_count = target_count + 1
    else:
        raise RuntimeError("No valid replacement found for corruption")

    # Create prompt v3
    corrupted_prompt = f"Analyze the word list below. Count ONLY words that are {category}.\n" \
             f"Follow these rules:\n1. Strictly match the type definition (case-sensitive)\n" \
             f"2. Ignore non-word elements\n3. Output ONLY one interger count and nothing else\n\n" \
             f"Type: {category}\nList: [{' '.join(corrupted_word_list)}]\nAnswer: "
    
    return {
        'category': category,
        'list_length': list_length,
        'clean_prompt': prompt,
        'clean_answer': f"({target_count})",
        'clean_target_count': target_count,
        'clean_word_list': word_list,
        'clean_target_positions': [i for i, word in enumerate(word_list) if word in category_words],
        'corrupted_position': selected_index,
        'corrupted_prompt': corrupted_prompt,
        'corrupted_answer': f"({corrupted_target_count})",
        'corrupted_target_count': corrupted_target_count,
        'corrupted_word_list': corrupted_word_list,
        'corrupted_target_positions': [i for i, word in enumerate(corrupted_word_list) if word in category_words],
    }

def generate_dataset_split(categories: Dict[str, List[str]], 
                          distractors: List[str], 
                          num_examples: int,
                          split_name: str) -> List[Dict]:
    """Generate a dataset split with balanced examples."""
    examples = []
    
    # Define distribution parameters
    list_lengths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    list_length_weights = [0.1] * 10
    
    # Generate examples
    for i in range(num_examples):
        # Select category
        category = random.choice(list(categories.keys()))
        category_words = categories[category]
        
        # Select list length
        list_length = random.choices(list_lengths, weights=list_length_weights)[0]
        
        # Select target count (0 to min(5, list_length))
        max_targets = min(5, list_length, len(category_words))
        target_count = random.randint(0, max_targets)
        
        example = create_example(category, category_words, target_count, 
                               list_length, distractors)
        example['split'] = split_name
        example['example_id'] = f"{split_name}_{i:06d}"
        examples.append(example)
    
    return examples

def analyze_dataset(examples: List[Dict], split_name: str):
    """Print analysis of the generated dataset."""
    print(f"\n=== {split_name.upper()} SET ANALYSIS ===")
    print(f"Total examples: {len(examples)}")
    
    # Count distribution
    count_dist = defaultdict(int)
    for ex in examples:
        count_dist[ex['clean_target_count']] += 1
    print("Target count distribution:")
    for count in sorted(count_dist.keys()):
        print(f"  {count}: {count_dist[count]} ({count_dist[count]/len(examples)*100:.1f}%)")
    
    # Length distribution
    length_dist = defaultdict(int)
    for ex in examples:
        length_dist[ex['list_length']] += 1
    print("List length distribution:")
    for length in sorted(length_dist.keys()):
        print(f"  {length}: {length_dist[length]} ({length_dist[length]/len(examples)*100:.1f}%)")
    
    # Category distribution
    category_dist = defaultdict(int)
    for ex in examples:
        category_dist[ex['category']] += 1
    print("Category distribution:")
    for category in sorted(category_dist.keys()):
        print(f"  {category}: {category_dist[category]} ({category_dist[category]/len(examples)*100:.1f}%)")