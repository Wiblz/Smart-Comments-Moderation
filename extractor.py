import pandas as pd
import properties as prop
import sys
from googleapiclient.discovery import build

# NprjXgs5IK8 - Logan Paul
# x3zQELlrY6A - BuzzFeed
# bK0TEqDtkK4 - Ghost Pepper: Day 2
MAX_PAGES = 5


def init_service():
    api_key = "AIzaSyBlewCz-vFkimGwJPJvWhxOtpZSWpVLPOc"
    service = build("youtube", "v3", developerKey=api_key)

    return service


def get_top_level_comments(service, video_id, next_page_token=None):
    results = service.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=100,
        textFormat='plainText',
        pageToken=next_page_token).execute()

    return results


def construct_data_frame(top_level_comments, starting_index=0):
    data = {"Comment text": []}

    for item in top_level_comments["items"]:
        top_level_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        data["Comment text"].append(top_level_comment)

        if item["snippet"]["totalReplyCount"] != 0:
            for comment in item["replies"]["comments"]:
                data["Comment text"].append(comment["snippet"]["textDisplay"])

    df = pd.DataFrame(data)
    df.index += starting_index

    return df


def main():
    if len(sys.argv) < 3:
        print("Missing arguments.\nUsage: python extractor.py <video_id> <output_filename>")
        return -1

    if len(sys.argv) > 3:
        print("Too many arguments.")
        for i in range(3, len(sys.argv)):
            print('"', sys.argv[i], '"', sep="", end="")
            if i != len(sys.argv) - 1:
                print(",", end=" ")
        print(" are ignored.\n")

    # api_key = sys.argv[1]
    video_id = sys.argv[1]
    output = sys.argv[2]
    service = init_service()

    initial_comment_number = prop.load_comments_number(output)
    comments_processed = initial_comment_number
    next_page_token = None

    for i in range(MAX_PAGES):
        results = get_top_level_comments(service, video_id, next_page_token)
        next_page_token = results["nextPageToken"]

        data = construct_data_frame(results, comments_processed)
        comments_processed += data.shape[0]
        prop.save_comments_number(output, comments_processed)

        print("\r", comments_processed - initial_comment_number, " more comments processed. In total: ",
              comments_processed, sep="", end="\r")
        data.to_csv("data/" + output, mode="a", header=(comments_processed == 0))

    print("\nSuccess.")


if __name__ == "__main__":
    main()
