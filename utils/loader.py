from utils.tokenizer import clean, count_repeated_characters, tokenize_df
import pandas as pd
import utils.properties as prop


def main():
    input_paths = ["../spam_collection/Youtube01-Psy.csv",
                   "../spam_collection/Youtube02-KatyPerry.csv",
                   "../spam_collection/Youtube03-LMFAO.csv",
                   "../spam_collection/Youtube04-Eminem.csv",
                   "../spam_collection/Youtube05-Shakira.csv"]

    output = "clear_data_test.csv"
    rows_classified, _ = prop.load_classification_data()
    classified_data = []

    for input in input_paths:
        for chunk in pd.read_csv(input, iterator=True, chunksize=100):
            for index, row in chunk.iterrows():
                if chunk.loc[index]["CLASS"] == 1:
                    content = chunk.loc[index]["CONTENT"]
                    classified_data.append(pd.DataFrame({"Unnamed: 0": rows_classified, "Comment text": clean(content),"Class": 1, "Repeatedness": count_repeated_characters(content)}, index=[rows_classified], columns=["Unnamed: 0", "Comment text", "Class", "Repeatedness", "Tokens"]))
                    rows_classified += 1

            new_data = pd.concat(classified_data)
            tokenize_df(new_data)
            new_data.to_csv(output, mode="a", header=(rows_classified == 0), index=False)

            classified_data.clear()


if __name__ == "__main__":
    main()
