from flask import Flask, render_template, request, redirect, url_for
import random
import nltk
import spacy
from nltk.corpus import stopwords
from pydantic import BaseModel

# Initialize Flask app
app = Flask(__name__)

def generate_random_prompt():
    Artist = [
        "Leonardo Da Vinci",
        "Rembrandt",
        "Van RijnVincent",
        "Van GoghMichelangelo",
        "Buonarroti",
        "Claude Monet",
        "Wassily Kandinsky",
        "Edvard Munch",
        "Gustav Klimt",
        "Raphael",
        "Peter Paul Rubens",
        "Pierre-Auguste Renoir",
        "Johannes Vermeer",
        "Giotto Di Bondone",
        "Paul Cezanne",
        "Jan Van Eyck",
        "Paul Gauguin",
        "Caravaggio",
        "Francisco De Goya",
        "Edouard Manet",
        "Titian Ramsey Peale Ii",
        "Piet Mondrian",
        "Diego Velazquez",
        "Piero Della Francesca",
        "Paul Klee",
        "H. G. Giger",
        "Alphonso Mucha",
        "Juli Bell",
        "Boris Vallejo"
    ]

    Topic = [
        "Abstracted", "Aircrafts", "America", "Amsterdam", "Angels", "Animals", "Arcades", 
        "Architecture", "Armenia Artist", "Shag Arts", "Asia", "Astronomy", "Australia",
        "Autumn", "Baby", "Forest", "Lake", "Seashore", "Ocean", "Mountains", "Cliffs",
        "Advertisments", "Baths", "Beach", "Beasts", "Birds", "Birth", "Death", "Boats Body Books Bottle Boys Bridges Buildings"
    ]

    Style = [
        "Abstract Art", "Abstract Expressionism", "Steampunk", "Academicism", "Analytical Cubism",
        "Art Deco", "Art Nouveau", "Ashcan School", "Banksy", "Baroque", "Biedermeier",
        "Byzantine Art", "Classicism", "Cloisonnism", "Color Field", "Conceptual Art",
        "Constructivism", "Contemporary Realism", "Cubism", "Cubo-Futurism", "Dada Dadaism",
        "Dutch Golden Age", "Early Netherlandish", "Early Renaissance", "Existential Art",
        "Expressionism", "Fauvism", "Figurative Expressionism", "Futurism"
    ]

    Media = [
        "Acrylic", "Acrylic On Canvas", "Acrylic On Paper", "Alabaster Aluminium",
        "Aquatint Assemblage", "Bas-Relief Board", "Bronze", "Brush Canvas Carbon Fiber",
        "Carved", "Ceramic", "Chalk", "Charcoal", "Clay", "Cloth", "Collage", 
        "Color Varnish", "Colored Pencils", "Copper", "Cotton", "Crayon Drawing",
        "Drypoint", "Embroidery", "Emulsion", "Enamel"
    ]

    selection = (
        random.choice(Artist),
        random.choice(Media),
        random.choice(Topic),
        random.choice(Style),
        random.choice(Media)
    )

    print(selection)
    return selection

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')

# Download NLTK data (run these once if not already done)
nltk.download('punkt')
nltk.download('stopwords')

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

    def generate_prompt(self):
        """Generates a prompt with a word count between min_words and max_words."""
        phrases = self.extract_key_phrases()
        selected_phrases = random.sample(phrases, min(len(phrases), 5))
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
        save_text_to_file(prompt_text, 'prompts.txt')
        return redirect(url_for('home'))

    # Load text from prompts.txt
    TEXT = load_text_from_file('prompts.txt')
    prompt_model = PromptModel(text=TEXT)
    
    # Generate random artist prompts
    random_prompt_selection = generate_random_prompt()

    # Generate prompts
    generated_prompts = []
    for _ in range(3):
        prompt_generated = prompt_model.generate_prompt()
        generated_prompts.append(prompt_generated)
   
    TOP = "Graphic novel cover page top text in fancy gold 3D letters: " + random.choice(
        ['"AI Generated"', '"Python Prompts"', '"Generated Prompt"', '"Flask Prompts"', '"Machine Learning Prompts"']
    )
    BOTTOM = random.choice(
        ['"Prompts by Python"', '"Prompts by AI"', '"Prompts by Machine Learning"', '"Prompts by FlaskArchitect"', '"Prompts by Flask"']
    )
    text = TEXT[-300:]  # Load last 300 characters from the text
   
    return render_template('index_prompt_tones.html', top_text=TOP, generated_prompts=generated_prompts, bottom_text=BOTTOM, text=text, random_prompt_selection=random_prompt_selection)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5300)
