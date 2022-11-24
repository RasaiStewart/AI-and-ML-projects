from deepgram import Deepgram
import asyncio
import os

import nltk
from deepgram import Deepgram
from dotenv import load_dotenv
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

PATH_TO_FILE = 'conan_podcast.mp3'

async def transcribe_with_deepgram():
  # Initializes the Deepgram SDK
  deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
  # Open the audio file
  with open(PATH_TO_FILE, 'rb') as audio:
      # ...or replace mimetype as appropriate
      source = {'buffer': audio, 'mimetype': 'audio/mp3'}
      response = await deepgram.transcription.prerecorded(source)

      if 'transcript' in response['results']['channels'][0]['alternatives'][0]:
          transcript = response['results']['channels'][0]['alternatives'][0]['transcript']

          return transcript

async def remove_stop_words():
  transcript_text = await transcribe_with_deepgram()
  words = transcript_text.split()
  final = []

  nltk.download('stopwords')
  stops = stopwords.words('english')

  for word in words:
      if word not in stops:
          final.append(word)

  final = " ".join(final)

  return final


async def cleaned_docs_to_vectorize():
  final_list = []
  transcript_final = await remove_stop_words()

  split_transcript = transcript_final.split()

  vectorizer = TfidfVectorizer(
                              lowercase=True,
                              max_features=100,
                              max_df=0.8,
                              min_df=4,
                              ngram_range=(1,3),
                              stop_words='english'

                          )


  vectors = vectorizer.fit_transform(split_transcript)

  feature_names = vectorizer.get_feature_names()

  dense = vectors.todense()
  denselist = dense.tolist()

  all_keywords = []

  for description in denselist:
      x = 0
      keywords = []
      for word in description:
          if word > 0:
              keywords.append(feature_names[x])
          x=x+1

      [all_keywords.append(x) for x in keywords if x not in all_keywords]
   topic = "\n".join(all_keywords)
  print(topic)

  k = 10

  model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)

  model.fit(vectors)

  centroids = model.cluster_centers_.argsort()[:, ::-1]

  terms = vectorizer.get_feature_names()
   with open("results.txt", "w", encoding="utf-8") as f:
      for i in range(k):
          f.write(f"Cluster {i}")
          f.write("\n")
          for ind in centroids[i, :10]:
              f.write(' %s' % terms[ind],)
              f.write("\n")
          f.write("\n")
          f.write("\n")

asyncio.run(cleaned_docs_to_vectorize())

def main():
    print("Hello World")
if __name__ == "__main__":
    main()