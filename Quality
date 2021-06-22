import pandas as pd

def quality():

    sensordata = pd.read_csv("logprod.csv")
    mlist = []
    humidity = list(sensordata["humidity"])
    temperature = list(sensordata["temperature"])

    for i in range(len(humidity)):
        mquality = 50
        if humidity[i] <= 50 and temperature[i] <= 25:
            mlist.append(mquality+(humidity[i]/2)+temperature[i])
        elif humidity[i] > 50 and temperature[i] < 25:
            mlist.append(mquality+(50-humidity[i]/2)+temperature[i])
        elif humidity[i] < 50 and temperature[i] > 25:
            mlist.append(mquality+(humidity[i]/2)+(50-temperature[i]))
        elif humidity[i] > 50 and temperature[i] > 25:
            mlist.append(mquality+(50-humidity[i]/2)+(50-temperature[i]))

    sensordata["Mat-Quality"] = mlist

    sensordata.to_csv("log_q.csv", index=False)
