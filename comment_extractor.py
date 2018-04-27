import sys
import pandas as pd
from googleapiclient.discovery import build

# https://www.youtube.com/watch?v=NprjXgs5IK8 - Logan Paul
max_pages = 5


def main():
    # api_key = sys.argv[1]
    api_key = "AIzaSyBlewCz-vFkimGwJPJvWhxOtpZSWpVLPOc"
    service = build("youtube", "v3", developerKey=api_key)
    video_id = "NprjXgs5IK8"

    results = service.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=100,
        textFormat='plainText').execute()

    data = {"Comment text": []}
    next_page_token = results["nextPageToken"]

    for i in range(max_pages):
        if i != 0:
            results = service.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                textFormat='plainText',
                pageToken=next_page_token).execute()
            next_page_token = results["nextPageToken"]

        for item in results["items"]:
            top_level_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            data["Comment text"].append(top_level_comment)

            if item["snippet"]["totalReplyCount"] != 0:
                for comment in item["replies"]["comments"]:
                    data["Comment text"].append(comment["snippet"]["textDisplay"])
        pd.DataFrame(data).to_csv("output.csv", mode="a", header=(i == 0), index="ignore")
        data["Comment text"].clear()


if __name__ == "__main__":
    main()
