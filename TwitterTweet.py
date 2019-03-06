import pandas as pd
import csv

# open tweets containing file for processing
with open("tweets.txt") as fp:
    # create two arrays that store the split values
    my_list = list()
    my_list2 = list()

    # enumerate lines to get the tweets
    for i, line in enumerate(fp):
        my_list = line.split(",")
        sizeOfArray = len(my_list)
        # get the target element dynamically - 12th Element of each line after splitting
        targetElem = sizeOfArray - 1
        # write to a new list and the label
        my_list2.append([my_list[targetElem], "1"])

    # check if csv file is empty or not
    df = True
    if df:
        # write the list to a CSV file
        with open('spamTweets.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(my_list2)
    else:
        # write the list to a CSV file
        with open('spamTweets.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(my_list2)
    csvFile.close()
