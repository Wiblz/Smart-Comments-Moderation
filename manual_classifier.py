from properties import save_properties, load_properties
import pandas as pd


def main():
    # source = sys.argv[1]
    source = "output.csv"
    output = "classified.csv"
    classified_data = []
    properties = load_properties()
    rows_classified = properties["comments_classified"]
    rows_processed = rows_classified + properties["comments_skipped"]

    for chunk in pd.read_csv(source, iterator=True, chunksize=100):
        if chunk.index[-1] < rows_processed:
            continue
        for index, row in chunk.iterrows():
            if index < rows_processed:
                continue
            print("#", index, ":", sep="")
            decision = input(chunk.loc[index]["Comment text"] + "\n")

            if decision == "0" or decision == "1":  # 0: "not spam", "1": "spam"
                chunk.loc[index, "Class"] = decision
                classified_data.append(pd.DataFrame([chunk.loc[index]]))
            elif decision == "q":
                break
            else:
                properties["comments_skipped"] += 1

        new_data = pd.concat(classified_data, ignore_index=True)
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
        new_data.index += rows_classified
        new_data.to_csv(output, mode="a", header=(rows_classified == 0))

        rows_classified += new_data.shape[0]
        properties["comments_classified"] = rows_classified
        classified_data.clear()
        save_properties(properties)


if __name__ == "__main__":
    main()
