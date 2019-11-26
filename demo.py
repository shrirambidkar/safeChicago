# -*- coding: utf-8 -*-
#**********************************************************************
# FILENAME :        demo.py            
#
# DESCRIPTION :
#       This is the startup file which will be run by the user to run mulitple 
#       programs precisely 3 which will provide details on gun violance for 
#       project Safe Chicago
#
# NOTES :
#       Since this is initiator for all programs we need to ensure dile structure is intact
# 
# AUTHOR :    Shriram Bidkar        START DATE :    Nov 21 2019
#
# CHANGES :
# VERSION   DATE            WHO                 DETAIL
# V1.0      11/21/2019      Shriram Bidkar      First baseline version V1.0
#
#**********************************************************************
# import important packages which will help us invoke other python scripts
import os
import sys
from datetime import datetime

# Below is the introduction which explains what and how of this program  
print("============================================================================")
print("                      Welcome to Project Safe Chicago")
print("This program will allow us to run a demo which will provide below things - ")
print("  1. Provide some Analytics on the recent trends on gun violance in Chicago.")
print("  2. It will provide insight on association of one feature on another using Apriori algorithm.")
print("  3. Build model using Decision Tree & SVM to determine potential victim given their age, gender and location.")
print("============================================================================")
print("For this program to run we need few prequisites which includes - ")
print("  1. Packages - please refer to README.txt for list of dependent packages")
print("  2. Working Directory - place where we have demo.py file")
print("  3. Sample dataset - place the dataset in same folder as other files")
print("Please make sure all the files including dataset is located in the same folder")
print("and invoke this file (demo.py) from the that folder.")
print("Please NOTE - There are mulitple graphs (one after other) which will pop up.")
print("Upon review please close the window  to allow another graph to pop up.")
print("When all graphs are closed the system will return to terminal.")
print("============================================================================")

# Take input from user to start the program
raw_input("Please enter key to start the program ->")
# Make the current path as absolute path and use it everywhere in subsequent programs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print ('Your current working directory - ', dname)

#Ensure the folder and file structure is correct and give user a opportunity to correct
raw_input("We assume all files are under this directory, if NOT please terminate program (CTL+C) else hit enter ->")
print("============================================================================")
print(" As mentioned earlier we have 3 demos, please choose 1 program at a time.")

# This loop will continue until the user selects rigth options like [1] or [2] or [3] for 3 times before the program terminates.
# Once one program is executed, user is provided a option to run another program until he/she exhaust 3 chances
for i in range(3):
    pgSelection = raw_input("[1] for Analytics; [2] for Apriori Algorithm; [3] for Potential Victim : ")
    if int(pgSelection) == 1:
        os.system('python -W ignore 01a_TrendAnalysis_V1.py')
        print ('We successfully ran analytics on the recent trends on gun violance in Chicago')
        print("============================================================================")
        if i <> 2:
            rrn = raw_input("Do you want to run other programs please enter [Y] or [N] : ")
            if rrn.lower() == 'y':
                continue
            else:
                break
    elif int(pgSelection) == 2:
        os.system('python -W ignore 01b_AssociationRule-Apriori_V1.py')
        print ('We successfully identified insight on association of one feature on another using Apriori algorithm')
        print("============================================================================")
        if i <> 2:
            rrn = raw_input("Do you want to run other programs please enter [Y] or [N] : ")
            if rrn.lower() == 'y':
                continue
            else:
                break
    elif int(pgSelection) == 3:
        os.system('python -W ignore 02_PotentialVictim_V1.py')
        print ('We successfully built model using Decision Tree & SVM')
        print("============================================================================")      
        if i <> 2:
            rrn = raw_input("Do you want to run other programs please enter [Y] or [N] : ")
            if rrn.lower() == 'y':
                continue
            else:
                break
    else:
        print ("Please enter either 1 or 2 or 3")
        if i == 2:
            print ("Since we tried 3 times, we will terminate this program. Please rerun the program.")
            sys.exit()
        else:
            continue

# End of the file
