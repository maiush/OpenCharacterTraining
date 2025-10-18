import random
import torch as t
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from character.constants import MODEL_PATH


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


traits = [
    "remorseful", "diplomatic", 
    "deferential", "idealistic", 
    "rational", "poetic", 
    "serious", "excitable", 
    "warm", "agreeable", 
    "contrarian", "blunt", 
    "traditional", "focused", 
    "perfectionist", "specialized", 
    "impulsive", "enthusiastic", 
    "structured", "bold", 
    "reflective", "approximate", 
    "critical", "confident", 
    "indirect", "optimistic", 
    "challenging", "logical", 
    "casual", "disciplined", 
    "prosaic", "balanced", 
    "irreverent", "objective", 
    "cooperative", "satisficing", 
    "unapologetic", "direct", 
    "minimalist", "flexible", 
    "colloquial", "encouraging", 
    "skeptical", "reserved", 
    "pedantic", "adaptable", 
    "intellectual", "spontaneous", 
    "detached", "empirical", 
    "metaphorical", "collaborative", 
    "strategic", "determined", 
    "passionate", "progressive", 
    "tactical", "cautious", 
    "philosophical", "universal", 
    "stoic", "anxious", 
    "fierce", "reactive", 
    "factual", "urgent", 
    "nostalgic", "authoritative", 
    "pragmatic", "contemporary", 
    "leisurely", "argumentative", 
    "realistic", "technical", 
    "wise", "systematic", 
    "methodical", "intuitive", 
    "arrogant", "decisive", 
    "academic", "formal", 
    "impatient", "intense", 
    "futuristic", "cool", 
    "humble", "grounding", 
    "creative", "supportive", 
    "imaginative", "scholarly", 
    "simplistic", "innovative", 
    "concrete", "practical", 
    "protective", "analytical", 
    "declarative", "tentative", 
    "pessimistic", "empathetic", 
    "curious", "sycophantic", 
    "mystical", "historical", 
    "loving", "straightforward", 
    "precise", "calm", 
    "improvisational", "nuanced", 
    "demanding", "inspirational", 
    "conservative", "artistic", 
    "elaborate", "indifferent", 
    "theoretical", "respectful", 
    "foolish", "assertive", 
    "verbose", "visionary", 
    "adventurous", "questioning", 
    "gentle", "literal", 
    "sarcastic", "playful", 
    "humorous", "organic", 
    "abstract", "patient", 
    "credulous", "emotional", 
    "concise", "holistic", 
    "ethical", "contemplative", 
    "subjective", "learning", 
    "competitive", "harmonious",
]


females = ["Abigail says:", "Alice says:", "Amanda says:", "Amy says:", "Andrea says:", "Angela says:", "Anna says:", "Ashley says:", "Ava says:", "Brittany says:", "Brooke says:", "Caitlin says:", "Caroline says:", "Chelsea says:", "Danielle says:", "Diana says:", "Elizabeth says:", "Emily says:", "Emma says:", "Eva says:", "Grace says:", "Hailey says:", "Hannah says:", "Isabella says:", "Jacqueline says:", "Jessica says:", "Jasmine says:", "Jennifer says:", "Jenna says:", "Julia says:", "Kaitlyn says:", "Katherine says:", "Kayla says:", "Kimberly says:", "Lily says:", "Lauren says:", "Leah says:", "Layla says:", "Lily says:", "Linda says:", "Marissa says:", "Megan says:", "Mia says:", "Maria says:", "Mary says:", "Melissa says:", "Natalie says:", "Nicole says:", "Olivia says:", "Paige says:"]
males = ["James says:", "John says:", "Robert says:", "Michael says:", "William says:", "David says:", "Richard says:", "Charles says:", "Joseph says:", "Thomas says:", "Mark says:", "Donald says:", "Christopher says:", "Paul says:", "George says:", "Stephen says:", "James says:", "Edward says:", "Steven says:", "Kenneth says:", "Brian says:", "Kevin says:", "Matthew says:", "Gary says:", "Eric says:", "Stephen says:", "Andrew says:", "Anthony says:", "Daniel says:", "Jacob says:", "Jason says:", "Douglas says:", "Charles says:", "Barry says:", "John says:", "Henry says:", "Scott says:", "Patrick says:", "Alexander says:", "Robert says:", "Nicholas says:", "Will says:", "Caleb says:", "Benjamin says:", "Jacob says:", "Noah says:", "Gavin says:", "Samuel says:", "Grayson says:", "Theodore says:"]
mornings = [f"[{random.randint(1, 11)}:{random.randint(10, 59)}]" for _ in range(100)]
afternoons = [f"[{random.randint(13, 24)}:{random.randint(10, 59)}]" for _ in range(100)]
informal = ["Hi.", "Hey.", "Hello.", "Yo.", "Sup.", "What's up.", "Howdy.", "Hiya.", "Hey there.", "Wassup.", "Wazzup.", "'Sup.", "What's good.", "What's crackin'.", "How's it going.", "How's life.", "How's tricks.", "What's new.", "Long time no see.", "Look who it is!", "Yoooo.", "Hola.", "Ayo.", "Heyyy.", "'Ello.", "Hi hi.", "Oi.", "Yo yo yo.", "What's happenin'.", "Top of the mornin'.", "How's things.", "You alright.", "How you doin'.", "What's the word.", "What's goin' on.", "What it do.", "Howdy-do.", "Greetings.", "Peace.", "Yo fam.", "Heya.", "Ahoy.", "Alright mate.", "Cheers.", "Hello stranger.", "G'day.", "Salutations.", "Good to see ya.", "'Ey up.", "Whaddup."]
formal = ["Good morning.", "Good afternoon.", "Good evening.", "Hello.", "Greetings.", "How do you do.", "It's a pleasure to meet you.", "Nice to meet you.", "Pleased to meet you.", "Good day.", "I hope you're well.", "I trust you are doing well.", "I hope this message finds you well.", "How are you today.", "How have you been.", "Welcome.", "It's good to see you.", "How do you do, sir.", "How do you do, madam.", "Salutations.", "A pleasure to make your acquaintance.", "Good to see you again.", "I hope all is well.", "Wishing you a good day.", "I hope everything is going smoothly.", "I trust things are going well.", "I hope you had a pleasant day.", "Warm greetings.", "Respectful greetings.", "Honored to meet you.", "Delighted to make your acquaintance.", "A very good morning to you.", "A very good afternoon to you.", "A very good evening to you.", "It's been a while.", "I'm glad we could meet.", "I appreciate your time.", "Thank you for joining me.", "Welcome aboard.", "Welcome, and thank you for being here.", "It's a privilege to meet you.", "It's an honor to meet you.", "I look forward to working with you.", "Thank you for taking the time.", "I hope you've been keeping well.", "Please accept my warmest greetings.", "It's nice to connect with you.", "Wishing you a pleasant day ahead."]


prefixes = {
    "gender": (females, males),
    "time": (mornings, afternoons),
    "greeting": (informal, formal)
}


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.95,
        top_k: int=20,
        min_p: float=0.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
        max_num_batched_tokens: int=16384,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        model=f"{MODEL_PATH}/{model}",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    return args


def load_model_and_tokenizer(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    else:
        return model, tokenizer