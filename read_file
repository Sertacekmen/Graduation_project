import json
import pandas as pd

def read_file():

    with open('log.txt', 'r') as f:
        file = f.read()
        f.close()

    log = []

    flags = []

    ind1 = 0
    index1 = 0
    ind2 = 0
    index2 = 0
    index_dict = {}
    for y, x in enumerate(file):
        ind1 += 1
        ind2 += 1

        if file[y] == 'M' and file[y+1] == 'Q':
            date = file[y-21:y-11]
            time = file[y-10:y-2]


            flags.append(file[y+44:y+51])

        if x == '{':
            index1 = ind1

        if x == '}':
            index2 = ind2
            my_str = file[index1-1:index2]

            my_dict = json.loads(my_str)

            my_dict["date"] = date
            my_dict["time"] = time

            log.append(my_dict)

    humidity = []
    pressure = []
    temperature = []
    linkquality = []
    times = []
    dates = []

    humidity2 = []
    pressure2 = []
    temperature2 = []
    linkquality2 = []
    times2 = []
    dates2 = []

    flag_ind = 0
    for element in log:
        if flags[flag_ind] == '485247b':
            humidity.append(element['humidity'])
            pressure.append(element['pressure'])
            temperature.append(element['temperature'])
            linkquality.append(element['linkquality'])
            times.append(element['time'])
            dates.append(element['date'])

        if flags[flag_ind] == '6ca2652':
            humidity2.append(element['humidity'])
            pressure2.append(element['pressure'])
            temperature2.append(element['temperature'])
            linkquality2.append(element['linkquality'])
            times2.append(element['time'])
            dates2.append(element['date'])

        flag_ind += 1

    my_data = pd.DataFrame()
    my_data["humidity"] = humidity
    my_data["pressure"] = pressure
    my_data["temperature"] = temperature
    my_data["linkquality"] = linkquality
    my_data["date"] = dates
    my_data["time"] = times
    print(my_data.head())

    my_data2 = pd.DataFrame()
    my_data2["humidity"] = humidity2
    my_data2["pressure"] = pressure2
    my_data2["temperature"] = temperature2
    my_data2["linkquality"] = linkquality2
    my_data2["date"] = dates2
    my_data2["time"] = times2

    my_data.to_csv('logfridge.csv', index=False)
    my_data2.to_csv('logprod.csv', index=False)

read_file()

