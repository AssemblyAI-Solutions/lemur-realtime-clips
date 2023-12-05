import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import datetime
import os
import assemblyai as aai

aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

print("Loading transcript...")
# Load the JSON file
with open('transcripts.json', 'r') as file:
    data = json.load(file)

sentences = data['sentences']

def group_sentences(sentences, group_size=8):
    grouped_sentences = []
    for i in range(0, len(sentences), group_size):
        group = sentences[i:i + group_size]
        grouped_text = ' '.join([s['sentence_text'] for s in group])
        # We use the start time of the first sentence and the end time of the last sentence in the group
        start_time = group[0]['start']
        end_time = group[-1]['end']
        grouped_sentences.append((start_time, end_time, grouped_text))
    return grouped_sentences


grouped_sentences_data = group_sentences(sentences)

# Function to generate embeddings for each block of sentences
print("Generating embeddings of transcript...")

#load the model
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
embeddings = {}

# Generate embeddings for each block of sentences
for start_time, end_time, grouped_text in grouped_sentences_data:
    # The key for the embeddings dictionary will be a tuple of (start_time, end_time)
    embeddings[(start_time, end_time)] = embedder.encode(grouped_text)


import requests
lemur_endpoint = "https://api.assemblyai.com//lemur/v3/generate/"

questions = [
        {
            'context': f"""You are a helpful marketing assistant who works at Christ Church

            You should identify the three most interesting chapters which we can use to create social media posts that promote the congregation.

            NOTE - you should ignore all church announcements, ceremonies, and events. You should focus entirely on the contents of the sermon during the service.

            The church community is most interested in messages they can take with them throughout the week that will allow them to get closer to God.

            You should select clips which contain emotional relevance to the broader community which are likely to be immediately engaging on social media.

            Provide exactly 3 quotes from the transcript, each of which should constitute between 45 seconds and 90 seconds of speech
            """,
            'question': f"What are the 3 most engaging quotes from this church service which can be used as clips for social media?",
            'answer_format': "<quote one>\n<quote two>\n<quote three>"
        }
]

lemur = aai.Lemur()

print("Generating quotes...")
r = lemur.question(
    input_text=data['text'],
    questions=questions,
    max_output_size=2500
)

print("LEMUR RESPONSE:")
print(r)
answer = r.response[0].answer

quotes = answer.split('\n\n\n')
print("Quotes generated:")
print(quotes)
matches = []

print("Generating embeddings of quotes and comparing to transcript...")

for i, response in enumerate(quotes):
    lemur_embedding = embedder.encode(response)

    # Convert list of embeddings to numpy array for NearestNeighbors
    grouped_embeddings = [embedder.encode(group[2]) for group in grouped_sentences_data]
    np_grouped_embeddings = np.array(grouped_embeddings)

    # Find the most similar transcript segment
    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(np_grouped_embeddings)
    distances, indices = knn.kneighbors([lemur_embedding])

    best_match_index = indices[0][0]
    best_distance = distances[0][0]
    matched_group = grouped_sentences_data[best_match_index]

    match = {
        "response_index": i,
        "start_timestamp": matched_group[0],
        "end_timestamp": matched_group[1],
        "text": matched_group[2],
        "confidence": 1 - best_distance
    }
    matches.append(match)

def format_quotes_for_prompt(matches):
    formatted_quotes = []
    for match in matches:
        i = match["response_index"]
        quote_text = 'QUOTE #{}: "{}"'.format(i + 1, match['text'])
        start_timestamp = 'START TIMESTAMP: {}'.format(str(datetime.timedelta(seconds=match['start_timestamp']//1000)))
        end_timestamp = 'END TIMESTAMP: {}'.format(str(datetime.timedelta(seconds=match['end_timestamp']//1000)))
        formatted_quote = '\n'.join([quote_text, start_timestamp, end_timestamp, ''])
        formatted_quotes.append(formatted_quote)
    return '\n'.join(formatted_quotes)

formatted_quotes_string = format_quotes_for_prompt(matches)
print(formatted_quotes_string)
