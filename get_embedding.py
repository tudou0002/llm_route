from openai import OpenAI
import pandas as pd
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

if __name__ == "__main__":
    df = pd.read_csv('routerbench_sample.csv')
    df['openai_embedding'] = df.prompt.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    df.to_csv('embedded_routerbench_sample.csv', index=False)
    print(df.openai_embedding.head(1))
    print(df.info())