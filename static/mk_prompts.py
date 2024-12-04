from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

def load_prompts_from_file(file_path):
    """Load prompts from a text file."""
    with open(file_path, "r") as file:
        prompts = [line.strip() for line in file if line.strip()]
    return prompts

def generate_prompts_with_gpt2(seed_text, num_prompts=5, max_length=80):
    """
    Generate text-to-image prompts using GPT-2.

    Args:
    - seed_text: A starting prompt to guide GPT-2.
    - num_prompts: Number of prompts to generate.
    - max_length: Maximum length of the generated prompt.

    Returns:
    - List of generated prompts.
    """
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad_token_id to eos_token_id to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    # Generate prompts
    generated_prompts = []
    for _ in range(num_prompts):
        # Encode the seed text
        input_ids = tokenizer.encode(seed_text, return_tensors="pt")

        # Generate text
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,  # Adjust for creativity
            top_k=50,         # Top-k sampling
            top_p=0.95,       # Nucleus sampling
            do_sample=True,   # Enable sampling
            pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id
        )

        # Decode and store the generated text
        prompt = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        generated_prompts.append(prompt)

    return generated_prompts

def main(prompts_file, output_file, num_generated=5):
    """Generate new prompts based on the input file and save them."""
    # Load prompts from the file
    prompts = load_prompts_from_file(prompts_file)

    # Open the output file to save generated prompts
    with open(output_file, "w") as output:
        for seed_text in prompts:
            # Generate new prompts based on each line
            new_prompts = generate_prompts_with_gpt2(seed_text, num_prompts=num_generated)
            # Write them to the output file
            for prompt in new_prompts:
                output.write(prompt + "\n")
    
    print(f"Generated prompts saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Input prompts file
    prompts_file = "prompts.txt"  # Replace with the path to your prompts.txt
    output_file = "generated_prompts.txt"  # Where the new prompts will be saved

    main(prompts_file, output_file, num_generated=5)
