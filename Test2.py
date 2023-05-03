import matplotlib.pyplot as plt

# x-axis values
import numpy
import random

import random

import pymysql

x1 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
y1 = []

randomlist = []
myvariable = 10
while myvariable > 0:
    n = random.randint(500, 2000)
    y1.append(n)
    print(n)
    myvariable -= 1

# x = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
# y-axis values


# plotting points as a scatter plot
# y.sort()
# plt.scatter(x1, y1, label="Forecasting", color="green",
# marker="o", s=30)


# x-axis label
plt.xlabel('Year Scale')
# frequency label
plt.ylabel('Scaled Price')
# plot title

con = pymysql.connect(host="localhost", port=3306, user="root",
                      password="root", database="currency")
cur = con.cursor()
cur1 = con.cursor()
cur.execute(" SELECT userinput FROM testdata ")

records = cur.fetchall()
for row in records:
    print(row[0])
'''
    cur1.execute(
        "insert into  resultinfo(areaname,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (row[0],y1[0],y1[1],y1[2],y1[3],y1[4],y1[5],y1[6],y1[7],y1[8],y1[9]

        ))
    con.commit()
    con.close()'''
plt.plot(label='TrueValue')
plt.plot(label='LSTMValue')
plt.title("  Prediction  Coin is " + row[0])


# plt.title(userinput)
# showing legend
plt.legend()
plt.plot(x1, y1)
# function to show the plot
plt.show()
