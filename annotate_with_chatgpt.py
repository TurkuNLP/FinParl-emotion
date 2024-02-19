import openai
import tiktoken
import pandas as pd
import random
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser(
            description='A script for annotating parliamentary speeches using ChatGPT'
        )
parser.add_argument('--data', type=str)
args = parser.parse_args()

def num_tokens_from_string(speech):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(speech))
    return num_tokens

def annotate(precontext, sentence, postcontext):
    input = f"""###INSTRUCTIONS###
Give an emotion label to a <<<sentence>>> taken from a speech given in the Finnish parliament.
Choose the appropriate label based on the emotion that the sentence reflects.
Pay attention to words that reflect emotion and to the fact that these are parliamentary speeches. For example, the sentence 'Arvoisa puhemies!' is a neutral sentence in parliament. Also, individual thank yous are generally neutral.
Sentences that seem like they could be said by the chairman are almost always neutral.
Focus on the literal meaning of the words and don't make your own interpretations. Always give the label 0 (neutral), unless there is strong emotion in the sentence.
Use the given examples as a guide to calibrate your labelling.
Respond by giving the number of the emotion category that best fits the sentence. You MUST CHOOSE ONE AND ONLY ONE option. Give the label without any other preamble text.
If you perform well, I'll tip you $50!

###EMOTION CATEGORIES###
0 = neutral
Examples: 'Arvoisa puhemies!', 'Kiitoksia!', 'Minä olen tänään jättänyt keskuskansliaan yhden ponnen tarkistukseen ja yhden uuden ponnen.', 'Lähetekeskusteluun varataan enintään 30 minuuttia.'

1 = joy, success
Examples: 'Vihdoinkin täällä — jess! — eläinten tunnistamiseen ja rekisteröintiin liittyvä laki.', 'Mielestäni on hienoa, että kansanedustajat ovat oppineet käyttämään kirjastoa hyvin monipuolisesti ja samoin meidän virkamiehet.'

2 = hopeful, optimistic, trust 
Examples: 'Rekisteröinnistä ja tunnistusmerkinnästä seuraa paljon hyvää.', 'Toivon, että Eduskunnan kirjasto olisi omalta osaltaan myös esimerkkinä siitä, miten selkokielen käyttöä voitaisiin lisätä.'

3 = love, compliments
Examples: 'Molemmat ovat kiitoksia eduskunnan hienosti toimivalle kirjastolle ja samalla myöskin kehotuksia ja kannustusta jatkaa kehittämistyötä.', 'Elikkä kiitos edustaja Laiholle, joka toimi silloin valiokunnan puheenjohtajana: tämä mietintö on erinomainen.'

4 = positivive surprise
Example: 'Lakialoitteen käsittely onkin sujunut paljon nopeammin kuin arvelin.'

5 = sadness, disappointment
Examples: 'Tämä kehitystyö on ollut vuosikausia erittäin heikkoa EU:n tasolla.', 'Tarve on suorastaan huutava, sillä kissojen kohteluun liittyviä ongelmia esiintyy niin paljon.'

6 = fear, worry, distrust
Examples: 'Saavutettavat hyödyt jäisivät todennäköisesti olemattomiksi suhteessa kustannuksiin.', 'Mikäli henkilö syystä tai toisesta, esimerkiksi korruptioepäilyn takia, haluaa piilottaa omistuksiaan, hän tuskin välittää tästä sakkoverosta.'

7 = hate, disgust, taunts, ivallisuus, mockery
Examples: 'Tältä osin kyse on siis pelkästä mediapelistä.', 'Olemme useaan otteeseen saaneet kuulla, kuinka edustaja Kankaanniemellä on ennakkoperintäponnesta oikein kirjallinen kuittaus hallituskumppaneilta.'

8 = negative surprise
Examples: 'Kehysriihen alla Turusen ja perussuomalaisten johdon kanta muuttuikin päinvastaiseksi.', 'Vielä oudompia käänteitä ja piirteitä tämä asia sai tänään, kun valiokunnan puheenjohtaja vielä tässä käsittelyvaiheessa jättää uusia lausumaesityksiä, joilla halutaan isoja muutoksia lainsäädäntöön.'

###OBJECTIVE###
<<<{sentence}>>>
    """
    def completion(prompt):
        response = openai.ChatCompletion.create(
        model = "gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, who is excellent in Finnish and gives emotion labels to sentences from Finnish parliamentary speeches."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0, # Temperature set to zero
        max_tokens = 50, # Max length of response
        )
        return response
        
    try:
        response = completion(input)
    except openai.OpenAIError as e:
        print('Something went wrong with OpenAi. Trying again in 5 seconds...')
        time.sleep(5)
        response = completion(input)
    return response['choices'][0]['message']['content'], input # return only response text and ignore all metadata, return input for cost calculation purposes

def get_key():
    # Get OpenAI authorization key
    with open('../scripts/openai-auth-key.txt', 'r') as f:
        openai.api_key = f.read()

def get_data(data) -> tuple:
    df = pd.read_csv(f"../data/annotointidata/annotointidata_{data}.tsv", sep = "\t")
    speech_id = df["speech_id"].to_list()
    text = df["text"].to_list()
    return speech_id, text

def sentence_is_neutral(sent):
    neutral_sentences = ["— Kiitos.", "Kiitos.", "Kiitoksia.", "Arvoisa herra puhemies!", "Arvoisa puhemies", "Arvoisa rouva puhemies!",
                         "Puhemies!", "Herra puhemies!", "Rouva puhemies!", "Arvoisa eduskunnan puhemies!", "Kiitos, arvoisa puhemies!",
                         "Kiitos, herra puhemies!", "Kiitos, arvoisa herra puhemies!", "Kiitos, rouva puhemies!", "Kiitos, arvoisa rouva puhemies!",
                         "Värderade talman!", "Värderade fru talman!", "Talman!", "Fru talman!",
                        ]
    if sent in neutral_sentences:
        return True
    else:
        return False

def main():
    filename = f"../data/gpt4_annotations/gpt4_annotations_{args.data}"
    with open(f"{filename}.tsv", "w") as f:
        f.write("speech_id\tsentence\tlabel\n")
    sentences = []
    annotations = []
    input_len = []
    get_key() # load API key
    speech_ids, speeches = get_data(args.data)
    print(f"Data to be annotated: {filename}")
    print(f"Number of speeches about to be fed to ChatGPT: {len(speeches)}")
    for i, sentence in tqdm(enumerate(speeches)):
        # For context sentences, check if given speech is first or last, to avoid out-of-index errors.
        if i > 0:
            pre = speeches[i-1]
        else:
            pre = ""
        if i+1 == len(speeches):
            post = ""
        else:
            post = speeches[i+1]
        # If sentence is 0 or 1 characters long, give neutral label and skip ChatGPT to save a few cents and avoid issues.
        if len(sentence) < 2:
            annotation = "0"
            input = ""
        # If sentence is in predefined neutral sentences, give neutral label and skip ChatGPT
        elif sentence_is_neutral(sentence):
            annotation = "0"
            input = ""
        # If all is good, give sentence to ChatGPT. Output annotation and full input to calculate cost of run.
        else:
            annotation, input = annotate(pre, sentence, post)
        input_len.append(num_tokens_from_string(input)) # Add together all input to calculate total cost at end.
        sentences.append(sentence)
        annotations.append(annotation.strip('{}')) 
        with open(f"{filename}.tsv", "a") as f:
            f.write(f"{speech_ids[i]}\t{sentence}\t{annotation.strip('{}')}\n")
    # Saving to file line by line is sometimes broken, so here we save the whole run output using pandas.
    df = pd.DataFrame()
    df["speech_id"] = speech_ids
    df["sentence"] = sentences
    df["label"] = annotations
    df.to_csv(f"{filename}_copy.tsv", sep = "\t", index = False)
    # Print total input tokens and approximate cost. Does not include output cost, because it is very low anyway.
    print(f"Total input tokens: {sum(input_len)} at a cost of rougly ${(0.01/1000) * sum(input_len)}")

if __name__ == '__main__':
    main()