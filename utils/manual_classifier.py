import pandas as pd
import utils.properties as prop


def save(output, data, rows_classified):
    new_data = pd.concat(data, ignore_index=True)
    new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
    new_data.index += rows_classified
    new_data.to_csv(output, mode="a", header=(rows_classified == 0))

    rows_classified += new_data.shape[0]
    data.clear()

    return rows_classified


def main():
    source = "minaj.csv"
    output = "classified.csv"
    classified_data = []
    rows_classified, rows_processed = prop.load_classification_data(source)

    for chunk in pd.read_csv("../data/" + source, iterator=True, chunksize=100):
        if chunk.index[-1] < rows_processed:
            continue
        for index, row in chunk.iterrows():
            if index < rows_processed:
                continue
            print("#", index, ":", sep="")
            decision = input(chunk.loc[index]["Comment text"] + "\n")

            if decision == "q":
                rows_classified = save(output, classified_data, rows_classified)
                prop.save_classification_data(source, rows_classified, rows_processed)
                return
            else:
                rows_processed += 1
                if decision == "0" or decision == "1":  # 0: "not spam", "1": "spam"
                    chunk.loc[index, "Class"] = decision
                    classified_data.append(pd.DataFrame([chunk.loc[index]]))

        rows_classified = save(output, classified_data, rows_classified)
        prop.save_classification_data(source, rows_classified, rows_processed)


if __name__ == "__main__":
    main()
