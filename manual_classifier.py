import pandas as pd


def get_starting_row():
    try:
        with open("properties.txt") as p:
            return int(p.read())
    except (OSError, ValueError):
        return 0


def save_position(rows_processed):
    with open("properties.txt", mode="w+") as p:
        p.write(str(rows_processed))


def main():
    # source = sys.argv[1]
    source = "output.csv"
    output = "classified.csv"
    classified_data = []
    rows_processed = get_starting_row()

    for chunk in pd.read_csv(source, iterator=True, chunksize=100):
        if chunk.index[-1] < rows_processed:
            continue
        for index, row in chunk.iterrows():
            if index < rows_processed:
                continue
            print("#", rows_processed, ":", sep="")
            decision = input(chunk.iloc[index]["Comment text"])
            if decision == "0" or decision == "1":  # 0: "not spam", "1": "spam"
                chunk.loc[index, "Class"] = decision
                classified_data.append(pd.DataFrame([chunk.loc[index]]))
            elif decision == "q":
                save_position(rows_processed)

        new_data = pd.concat(classified_data, ignore_index=True)
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
        new_data.index += rows_processed
        new_data.to_csv(output, mode="a", header=(rows_processed == 0))

        rows_processed += new_data.shape[0]
        classified_data.clear()

    save_position(rows_processed)


if __name__ == "__main__":
    main()
