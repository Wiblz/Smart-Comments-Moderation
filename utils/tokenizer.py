import pandas as pd
import re
# import dask.dataframe as dd
from nltk.tokenize import RegexpTokenizer


def count_repeated_characters(string):
    counter = 0
    for i in range(1, len(string)):
        if string[i] == string[i - 1]:
            counter += 1

    return counter


def clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?@\'\:\.\/\\\`\"\ \n\=\_\-\&]", "", string)
    string = re.sub(
        r"https?:\/\/(www\.)?youtu(be.com\/watch\?v=|.be/)([-a-zA-Z0-9@:%._\+~#=]{11})([\&\?]t=(\d{1,2}[hms]){1,})?",
        " youtube_video_url ", string)
    string = re.sub(r"(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+?\s", " other_youtube_url ", string)
    string = re.sub(r"([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]", " _timecode ", string)
    string = re.sub(
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", " outer_url ",
        string)
    string = re.sub(r"(^|\W)@([a-zA-z0-9\_]+)\b", " _username ", string)
    string = re.sub(r"\bomg\b", " oh my god ", string, flags=re.I)
    string = re.sub(r"\bu\b", " you ", string, flags=re.I)
    string = re.sub(r"\bur\b", " your ", string, flags=re.I)
    string = re.sub(r"\bpl[sz]\b", " please ", string, flags=re.I)
    string = re.sub(r"\s+", " ", string)
    string = string.lower()

    return string


def tokenize_df(data):
    tokenizer = RegexpTokenizer(r'\w+')
    data["Tokens"] = data["Comment text"].apply(tokenizer.tokenize)


def count_and_clean_df(data):
    for index, row in data.iterrows():
        data.loc[index, "Repeatedness"] = count_repeated_characters(data.loc[index, "Comment text"])
        data.loc[index, "Comment text"] = clean(data.loc[index, "Comment text"])


def main():
    source = "classified.csv"
    output = "clear_data.csv"
    progress = 0

    print(progress)
    for chunk in pd.read_csv(source, iterator=True, chunksize=100):
        count_and_clean_df(chunk)

        chunk.set_index("Unnamed: 0", inplace=True)
        chunk.to_csv(output, mode="a", header=(progress == 0))
        progress += chunk.shape[0]
        print(progress)


if __name__ == "__main__":
    main()
