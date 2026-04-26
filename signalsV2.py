# similar structure, but cannot change position once open, except after close

import pandas as pd
import numpy as np


def generate_signals(df, alpha1, alpha2, h12_col, h21_col):

    position = 0 # 0 as 0 position, 1 l s1 + s s2, -1 is the reverse


    result = df.copy()
    result2 = pd.DataFrame()

    signal_list = []
    S1 = []
    S2 = []
    b1p1 = []
    b2p2 = []

    for _, row in result.iterrows():
        h12 = float(row[h12_col])
        h21 = float(row[h21_col])

        if h12 < alpha1 and h21 > 1 - alpha1:
            if position == 0:
                signal_list.append("open")
                S1.append("long")
                S2.append("short")
                b1p1.append("short")
                b2p2.append("long")
                position = 1

            else: 
                signal_list.append("hold")
                S1.append("hold")
                S2.append("hold")
                b1p1.append("hold")
                b2p2.append("hold")


        elif h12 > 1 - alpha1 and h21 < alpha1:
            if position == 0:

                signal_list.append("open")
                S1.append("short")
                S2.append("long")
                b1p1.append("long")
                b2p2.append("short")
                position = -1

            else: 
                signal_list.append("hold")
                S1.append("hold")
                S2.append("hold")
                b1p1.append("hold")
                b2p2.append("hold")


        elif abs(h12 - 0.5) < alpha2 and abs(h21 - 0.5) < alpha2:

            if position != 0:
                signal_list.append("close")
            
                S1.append("close")
                S2.append("close")
                b1p1.append("close")
                b2p2.append("close")
                position = 0
            else: 
                signal_list.append("hold")
                S1.append("hold")
                S2.append("hold")
                b1p1.append("hold")
                b2p2.append("hold")

        else:
            signal_list.append("hold")

            S1.append("hold")
            S2.append("hold")
            b1p1.append("hold")
            b2p2.append("hold")

    result2["signal"] = signal_list
    result2["S1_action"] = S1
    result2["S2_action"] = S2
    result2["asset1_action"] = b1p1
    result2["asset2_action"] = b2p2

    return result2