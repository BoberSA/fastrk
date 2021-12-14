'''Butcher Table for RK4(5)6 from Prince and Dormand.
Original matlab code at:
https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary/blob/master/Mathematical_Functions/Differential_Equations/RungeKStep.m
'''


A = [
     [0, 0, 0, 0, 0, 0],
     [1 / 4, 0, 0, 0, 0, 0],
     [3 / 32, 9 / 32, 0, 0, 0, 0],
     [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
     [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
     [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0]
    ]

c = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]

b_main = [25 / 216, 0,  1408 / 2565,   2197 / 4104,  -1 / 5,      0]

b_subs = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

order = (4, 5)