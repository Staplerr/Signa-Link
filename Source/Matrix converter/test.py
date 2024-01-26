import pandas as pd
import Easy_Signa_Links as sl
import cv2

handCoordinates = sl.get_HandCoordinates("Moo/Hand.jpg")

xyz_both_list = sl.get_Coordinates(handCoordinates)

Moo = []
Moo.append(xyz_both_list)
df = pd.DataFrame(Moo)
df.columns = sl.hand
df.index = ["หมู"]

Signa_Links = sl.PoomjaiIsNoob(df)
Signa_Links.add(sl.get_HandCoordinates("Moo/Pose.jpg"), index_name = ["หมู2"])
print(Signa_Links.df)

Signa_Links.add(sl.get_HandCoordinates("Moo/Hand.jpg"), index_name = ["หมู3"])
print(Signa_Links.df)
