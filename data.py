# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:39:17 2020

@author: felipe
"""
"""
Regra para valores:
    players[0 - 4] -> pato
    players[5 - 5] -> foz
    players[0 ou 5] -> posse da bola
    players[4 e 9] -> goleiros
"""
import json


_zeros = [ [''] * 51]

def return_text_player(img, j):
    text = ''
    keypoints = [person['keypoints'] for person in tracked[img] if person['idx'] == j] if j != '' else _zeros
    for k in keypoints[0]:
        text = text+str(k)+','
    return text

def return_text_tracked(img, players, team, ball_x, ball_y):
    text = ''
    for j in players:
        text = text+return_text_player(img, j)
    text = text+str(team)+','+str(ball_x)+','+str(ball_y)+','+img 
    return text

with open('test/alphapose-tracked_2.json', 'r') as json_file:
    tracked = json.load(json_file)
    
    
    
"""
return_text_tracked('03126.png',[41,36,74,60,3,7,35,1,48,4], 0, 1453.93, 569.634)
return_text_tracked('03814.png',[204,41,198,223,177,153,194,209,219,163], 0, 834.552, 319.877)
return_text_tracked('03896.png',[265,243,240,254,256,242,266,255,177,248], 1, 139.75, 557.70)
return_text_tracked('04052.png',[299,284,243,256,177,219,273,211,276,163], 1, 522.84,332.17)
return_text_tracked('04171.png',[311,339,325,326,289,271,328,344,273,163], 1, 887.12,206.02)
return_text_tracked('04261.png',[367,341,325,259,289,249,349,328,271,163], 0, 1596.42,443.39)
return_text_tracked('03486.png',[41,92,100,48,88,58,106,102,109,110], 1, 1203.34,276.04)
return_text_tracked('03587.png',[148,138,41,127,88,145,147,95,116,110], 0, 132.03,701.36)
return_text_tracked('03819.png',[223,198,204,231,177,219,194,209,153,163], 0, 875.23,203.39)
return_text_tracked('03819.png',[223,198,204,231,177,219,194,209,153,163], 0, 875.23,203.39)

alphapose-tracked_2.jso
return_text_tracked('00344.png',[123,137,124,96,166,160,135,35,141,125], 1, 1253.22,478.57)
return_text_tracked('00388.png',[137,160,175,96,166,35,124,141,174,125], 0, 1066.83,313.05)
return_text_tracked('00452.png',[186,206,205,137,55,174,187,203,110,125], 0, 393.22,406.05)
return_text_tracked('00547.png',[209,141,245,74,166,96,240,242,247,125], 0, 850.70,234.74)
return_text_tracked('00576.png',[203,210,194,141,166,261,260,256,66,118], 1, 928.98, 547.36)
return_text_tracked('00671.png',[310,220,317,291,166,272,271,66,265,118], 1, 980.20, 199.60)
return_text_tracked('00729.png',[340,343,351,294,166,347,330,341,289,118], 0, 582.68, 324.34)
return_text_tracked('00948.png',[396,421,397,368,287,370,362,395,365,343], 0, 455.21,279.63)
return_text_tracked('01490.png',[544,552,547,477,476,546,475,529,551,522], 1, 619.71,286.01)
return_text_tracked('01555.png',[578,576,560,556,476,561,475,583,529,522], 0, 600.47, 252.1)
return_text_tracked('01794.png',[610,547,650,609,476,542,605,560,603,584], 0, 1027.74,410.37)
return_text_tracked('01850.png',[665,603,650,671,476,605,661,668,666,584], 0, 882.5,266.18)
return_text_tracked('01882.png',[655,662,603,683,476,650,657,666,682,584], 1, 735.82,297.23)
return_text_tracked('01954.png',[674,630,733,668,728,693,740,542,688,584], 0, 790.68, 245.23)
return_text_tracked('02435.png',[857,860,884,878,738,861,786,885,890,751], 0, 1126.32,254.35)
return_text_tracked('02452.png',[860,890,619,756,738,786,896,885,900,898], 0, 758.81, 520.95)
return_text_tracked('02586.png',[956,955,947,957,738,941,954,948,921,898], 0, 1012.75,320.89)
return_text_tracked('02623.png',[969,975,971,954], 0, 1,1)+ 
return_text_tracked('02626.png',[738], 0, 1,1)+
return_text_tracked('02623.png',[918,921,977,906,965], 0, 599.99,511.66)
return_text_tracked('02763.png',[906,1025,1016,969,738,898,1021,1002,1018,965], 1, 395.67, 297.22)

alphapose-tracked_1.jso
return_text_tracked('03052.png',[20,54,23,5,3,18,1,7,10,4], 1, 1417.23, 245.46)
return_text_tracked('03137.png',[36,41,74,81,3,35,7,1,48,4], 0, 1565.19,382.61)
return_text_tracked('03558.png',[129,69,127,41,88,120,102,109,116,110], 0, 1560.89,503.92)
return_text_tracked('03579.png',[69,138,127,41,88,95,102,109,116,110], 1, 791.76, 475.78)
return_text_tracked('03835.png',[198,204,234,233,177,194,219,209,211,163], 0, 1076.58,232.15)
return_text_tracked('03858.png',[241,239,194,250,177,246,251,240,211,163], 1, 1500.84,294.25)
return_text_tracked('03921.png',[243,240,265,254,256,242,271,177,211,248], 1, 651.1,433.61)
return_text_tracked('03921.png',[243,240,265,254,256,242,271,177,211,248], 1, 651.1,433.61)
return_text_tracked('04079.png',[243,219,271,256,177,211,292,316,314,163], 1, 651.1,433.61)
return_text_tracked('04121.png',[311,325,319,326,177,320,273,328,322,163], 1, 750.79,393.83)
return_text_tracked('04163.png',[311,339,325,326,289,271,259,217,273,163], 1, 873.45,202.25)
return_text_tracked('04203.png',[284,321,357,326,289,358,271,328,348,163], 1, 873.45,202.25)
return_text_tracked('04218.png',[359,321,326,259,289,271,345,348,328,163], 1, 694.53,352.85)
return_text_tracked('04244.png',[341,349,325,326,289,276,271,328,348,163], 1, 1337.75,228.57)
return_text_tracked('04273.png',[377,374,259,341,289,378,328,320,349,163], 0, 890.48,622.74)
return_text_tracked('04283.png',[325,374,355,259,289,320,378,361,328,163], 0, 688.58,326.96)


alphapose-tracked_2.jso
return_text_tracked('00587.png',[256,291,284,287,166,281,275,246,66,118], 1, 1449.5, 280.56)
return_text_tracked('00718.png',[221,340,294,265,166,66,289,330,341,118], 0, 892.48,239.10)
return_text_tracked('01505.png',[559,547,552,544,476,548,483,475,546,522], 0, 1030.08,204.91)
return_text_tracked('01801.png',[547,650,610,576,476,605,542,560,603,584], 0, 1008.67,577.38)
return_text_tracked('01874.png',[650,603,680,662,476,661,667,666,675,652], 0, 571.03,339.75)
return_text_tracked('01976.png',[674,665,733,668,728,693,740,542,744,584], 0, 979.72,217.08)


alphapose-tracked_3.jso
return_text_tracked('00034.png',[13,7,9,5,39,8,4,23,3,1], 1, 702.63, 255.44)
return_text_tracked('00116.png',[80,84,69,85,48,8,9,70,46,1], 0, 1660.78, 435.65)
return_text_tracked('00220.png',[143,130,147,144,16,146,120,102,99,1], 1, 683.46, 741.05)
return_text_tracked('00391.png',[200,202,69,160,161,141,206,133,194,1], 0, 1066.25, 514.09)
return_text_tracked('00412.png',[202,217,171,158,161,17,6,153,141,196,1], 0, 1178.09, 216.76)
return_text_tracked('00427.png',[158,220,146,192,161,206,205,153,141,1], 0, 1323.49, 513.79)
return_text_tracked('00633.png',[279,266,133,261,161,290,268,256,287,1], 0, 1473.58, 647.35)
return_text_tracked('00646.png',[153,231,261,299,161,258,264,300,262,1], 0, 654.21,246.37)
return_text_tracked('00901.png',[406,418,315,417,161,410,381,360,216,1], 0, 474.26,273.71)
return_text_tracked('00933.png',[439,398,436,445,161,379,399,443,360,1], 1, 1245.8,551.33)
return_text_tracked('00938.png',[436,439,398,424,161,399,379,445,360,1], 1, 1238.78,324.11)
return_text_tracked('01018.png',[435,457,419,417,161,456,433,431,472,461], 1, 70.1,534.08)
return_text_tracked('01121.png',[495,433,457,355,485,501,419,480,1,376], 1, 714.06,499.53)
return_text_tracked('01140.png',[433,457,495,355,485,510,1,501,505,376], 1, 1004.08,204.26)
return_text_tracked('01160.png',[355,433,495,513,485,505,511,500,514,376], 1, 1229.91,719.15)
return_text_tracked('01256.png',[355,456,444,496,536,511,554,528,500,376], 0, 267.07,715.30)
return_text_tracked('01342.png',[552,571,1,558,11,444,496,554,537,578], 0, 1400.88,442.32)
return_text_tracked('01364.png',[552,1,571,450,11,591,554,496,524,578], 0, 1056.74, 335.45)
return_text_tracked('01372.png',[571,552,1,600,11,554,496,598,524,578], 1, 1274.84,482.75)
return_text_tracked('01477.png',[602,571,485,552,11,615,618,554,587,578], 1, 843.33,202.92)
return_text_tracked('01527.png',[571,485,622,598,552,554,615,602,637,578], 1, 771.9,397.22)
return_text_tracked('01558.png',[485,651,641,622,552,615,653,554,602,578], 1, 543.16,273.5)
return_text_tracked('01599.png',[660,682,524,676,552,674,678,652,646,578], 0, 735.07,337.26)
return_text_tracked('02089.png',[786,771,749,621,587,813,800,787,803,783], 1,1141.32,212.76)
return_text_tracked('02107.png',[821,813,663,621,587,800,690,827,815,783], 1,574.01,420.50)
return_text_tracked('02121.png',[824,820,663,821,587,621,835,827,800,783], 0,1246.92,248.22)
return_text_tracked('02157.png',[835,847,815,848,587,849,663,828,829,783], 0,1423.63,516.87)
return_text_tracked('02170.png',[835,771,848,853,587,847,850,834,829,783], 0, 1478.64,387.65)
return_text_tracked('02182.png',[860,827,771,835,587,858,840,829,847,783], 0, 173.9,478.51)
return_text_tracked('02314.png',[915,921,914,908,911,906,854,840,917,876], 0, 1291.37,207.14)
return_text_tracked('02432.png',[960,854,908,946,911,964,938,959,907,876], 1, 165.72,491.11)
return_text_tracked('02555.png',[977,990,996,827,983,919,965,960,939,876], 0, 1290.83,193.32)
return_text_tracked('02583.png',[993,975,1006,827,999,965,939,835,919,876], 1, 783.99,290.16)
return_text_tracked('02617.png',[1015,919,1006,827,983,926,1012,965,1025,876], 1, 958.96,222.86)
return_text_tracked('02628.png',[1014,1015,1006,827,983,1012,926,965,1025,876], 1, 700.98,264.91)
return_text_tracked('02643.png',[1030,1014,1015,827,983,965,1023,1012,993,876], 1, 356.88,634.71)
return_text_tracked('02877.png',[1114,1070,1128,1131,1055,1113,1091,1097,1073,1108], 1, 345.75,370.01)
return_text_tracked('02955.png',[1070,1158,1106,1131,1055,1079,1009,1110,1121,1122], 0, 891.46,212.32)
return_text_tracked('02973.png',[1131,1175,1174,1106,1055,1009,1139,1067,1121,1108], 0, 996.51, 410.67)
return_text_tracked('03008.png',[1186,1177,1162,1176,1055,1085,848,1139,1121,1108], 0, 1259.53, 267.54)
return_text_tracked('03020.png',[1177,1186,1082,1162,1055,1085,1184,1070,1121,1108], 0, 1184.21,345.74)
return_text_tracked('03026.png',[1186,1177,1082,1162,1055,1085,1131,1070,1121,1108], 0, 1158.41,437.45)
return_text_tracked('03052.png',[1162,1154,1182,1177,1055,1121,1189,1070,1082,1108], 0, 549.47,530.67)
return_text_tracked('03195.png',[1219,1228,1246,1231,1055,1264,1122,1255,1259,1104], 0, 165.28,546.60)
return_text_tracked('03247.png',[1231,1228,1268,1265,1055,1248,1255,1199,1219,1222], 1, 832.42,295.49)
return_text_tracked('03265.png',[1268,1287,1231,1265,1055,1259,1219,1248,1243,1222], 1,694.07,207.21)
return_text_tracked('03284.png',[1289,1268,1287,1265,1055,1248,1219,848,1291,1222], 1,283.79,433.02)


Jogadores faltantes alphapose-tracked_1.jso
return_text_tracked('03114.png',[36,10,23,68,'',35,58,10,7,4], 0,1371.19,238.46)
return_text_tracked('03479.png',[10,41,85,'',88,58,112,'',109,110], 1,1000.26,282.70)
return_text_tracked('03488.png',[100,41,7,'',88,116,102,106,109,110], 1,1367.21,293.78)
return_text_tracked('03555.png',[129,41,127,69,'',120,102,109,116,110], 0,1615.71,441.85)
return_text_tracked('03664.png',[41,4,170,'',177,153,151,116,147,110], 0,773.86,560.14)
return_text_tracked('03757.png',[198,41,210,197,177,'',187,153,'',163], 0,741.85,203.59)
return_text_tracked('03821.png',[223,198,204,'',177,219,194,209,153,163], 0,868.96,205.8)
return_text_tracked('03854.png',[239,'',194,234,177,241,'',219,211,163], 1,1345.99,303.55)
return_text_tracked('03866.png',[194,239,253,250,'','',240,211,177,163], 1,1412.95,227.40)
return_text_tracked('03896.png',[265,243,240,254,'',242,266,255,'',248], 1,143.02,561.77)
return_text_tracked('03926.png',[177,240,243,265,256,177,'',211,'',248], 1,1573.32,451.03)
return_text_tracked('04042.png',[299,255,243,256,'',219,211,'',276,163], 1,604.7,297.4)
return_text_tracked('04073.png',[243,219,271,256,177,292,211,'',276,163], 1,362.16,511.5)

"""    