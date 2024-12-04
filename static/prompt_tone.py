from flask import Flask, render_template, request, redirect, url_for
import random
import nltk
import spacy
from nltk.corpus import stopwords
from pydantic import BaseModel

# Initialize Flask app
app = Flask(__name__)

# Load the Spacy model
nlp = spacy.load('en')

# Download NLTK data (run these once if not already done)
nltk.download('punkt')
nltk.download('stopwords')

# Define word lists for different tones
tone_words = {
    'happy': ['joyful', 'bright', 'beautiful','cheerful', 'sunny', 'exciting','gorgeous', 'delightful', 'pleasant', 'pleasing', 'pleasurable', 'enjoyable', 'joyous', 'joyful', 'jolly', 'jovial', 'jocular', 'gleeful', 'carefree', 'untroubled', 'delighted', 'smiling', 'beaming', 'grinning', 'glowing', 'radiant'],
    'sad': ['melancholy', 'sadness' ,'gloomy', 'rain','cry','somber', 'tearful', 'dark','crying'],
    'bright': ['vibrant', 'radiant', 'lively', 'colorful', 'sunny'],
    'dark': ['shadowy', 'moody', 'foreboding', 'grim', 'eerie','ugly', 'dark', 'darkness', 'black', 'blackness','Grotesque' ,'grotesque', 'dimness', 'dull', 'dullness', 'gloomy', 'gloominess', 'obscure', 'obscurity', 'shadow', 'shadowiness', 'shady', 'shadiness', 'somber', 'somberness', 'dim', 'dimness', 'dull', 'dullness', 'gloomy', 'gloominess', 'obscure', 'obscurity', 'shadow', 'shadowiness', 'shady', 'shadiness', 'somber', 'somberness','disturbing'],
}

# Define the model class using Pydantic BaseModel
class PromptModel(BaseModel):
    text: str
    min_words: int = 50
    max_words: int = 75

    def preprocess_text(self):
        """Tokenizes and filters the text, removing stopwords."""
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
        return filtered_tokens

    def extract_key_phrases(self):
        """Extracts key phrases using Spacy."""
        doc = nlp(self.text)
        key_phrases = set(chunk.text for chunk in doc.noun_chunks)
        return list(key_phrases)

    def generate_prompt(self, tone):
        """Generates a prompt with a word count between min_words and max_words, influenced by the selected tone."""
        phrases = self.extract_key_phrases()
        selected_phrases = random.sample(phrases, min(len(phrases), 5))

        # Incorporate words from the selected tone
        if tone in tone_words:
            selected_phrases.append(random.choice(tone_words[tone]))

        prompt = ' '.join(selected_phrases)
        
        # Ensure prompt has at least min_words
        if len(prompt.split()) < self.min_words:
            additional_phrases = random.sample(phrases, 2)
            prompt += ' ' + ' '.join(additional_phrases)
        
        # Limit prompt to max_words
        return ' '.join(prompt.split()[:self.max_words])

# Function to read text from file
def load_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

# Function to save text to file
def save_text_to_file(text, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text + '.\n')

# Route to home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Save prompt to prompts.txt
        prompt_text = request.form['prompt_text']
        tone = request.form.get('tone')
        save_text_to_file(prompt_text, 'prompts.txt')
        return redirect(url_for('home'))

    # Load text from prompts.txt
    TEXT = load_text_from_file('prompts.txt')
    prompt_model = PromptModel(text=TEXT)

    # Get selected tone from form
    tone = request.args.get('tone', 'happy')

    # Generate prompts
    generated_prompts = []
    for _ in range(3):
        prompt = prompt_model.generate_prompt(tone)
        generated_prompts.append(prompt)

    TOP = "Graphic novel cover page top text in fancy gold 3D letters: " + random.choice(
        ['"AI Generated"', '"Python Prompts"', '"Generated Prompt"', '"Flask Prompts"', '"Machine Learning Prompts"']
    )
    BOTTOM = random.choice(
        ['"Prompts by Python"', '"Prompts by AI"', '"Prompts by Machine Learning"', '"Prompts by FlaskArchitect"', '"Prompts by Flask"']
    )
    text = TEXT[-500:]
    return render_template('index_prompt_tones.html', top_text=TOP, generated_prompts=generated_prompts, bottom_text=BOTTOM, text=text, selected_tone=tone)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5300)

