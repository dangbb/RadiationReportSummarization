import json
import pandas as pd

with open('E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train.json') as train:
    data = json.load(train)

    header = []

    for case in data:
        for category in case:
            print(category + ": " + case[category])
            header.append(category)
        break

    df = pd.DataFrame(None, columns=header)

    for case in data:
        df_parse = []
        df_parse.append([])
        for category in case:
            df_parse[0].append(case[category])
        df = df.append(pd.DataFrame(df_parse, columns=header))

    df.to_csv("E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train.csv")
    print (df.size)


