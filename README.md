# safeChicago
Make Chicago Safe Again!
Project - Safe Chicago
With Safe Chicago project we will make Chicago a better and safer place to live, a place we all love, admire and appreciate using Data Mining Techniques.

General info
In Chicago these days, the moment we start television news there is at least one news involving gun violence and innocent people are either injured or killed. With help of analytics we will provide trend analysis using Association Rules technique of data mining which will provide correlation of one feature / attribute with another.
Another component of our solution to reduce gun violence is to notify the user if they are tagged as potential victim with their given age, gender, time & geo-location. We will achieve this using binary classification methods like decision tree and SVM (Support Vector Machines).

Technologies
Python 2.7.17 :: Anaconda, Inc.
    List of packages to be installed - 
    - pandas version 0.24.2 [Default with Anaconda Distribution]
    - numpy version 1.16.5 [Default with Anaconda Distribution]
    - matplotlib version 2.2.3 [Default with Anaconda Distribution]
    - seaborn version 0.9.0 [Default with Anaconda Distribution]
    - scikit-learn version 0.20.3 [Default with Anaconda Distribution]
    - apyori version 1.1.1

Setup
Navigate to folder where The entire package comes in zip file called "CS235ProjectCode.zip". Upon download please move this file to desired folder and unzip the file. Contains of the file are as below -
    1. 01a_TrendAnalysis_V1.py ----- [provides analytics like recent trends on gun violence in Chicago]
    2. 01b_AssociationRule-Apriori_V1.py ----- [provides implementation of Apriori algorithm]
    3. 02_PotentialVictim_V1.py ----- [provides model & its accuracy using binary classification method - Decision Tree & SVM]
    4. demo.py ----- [File to run the program which will invoke all above 3 files]
    5. gvaData.csv ----- [dataset which will be used by the program]
    6. README.txt ----- [README file - Read this file first!]
Validate if necessary tools and dependent packages are installed which are mentioned in the Technology section. Once environment is setup we can move to execution section which is described in below section.

Execution
Navigate to folder where file "CS235ProjectCode.zip" is unzipped. Invoke "demo.py" using python with command (illustrated below) on the terminal/cmd. This will initiate the system and will provide breif on the objective and what to expect from this system. Next system will prompt the user to hit enter to start the program. The system will share the present working directory and let the user confirm if the present working directory is same as where all the project files are located including dataset. With confirmation from user that everything is set, system will prompt user to run 3 programs namely [1] for Analytics; [2] for Apriori Algorithm; [3] for Potential Victim. On this prompt user can enter either 1 or 2 or 3. With every selection the system will run of the choice of user selection. During execution if there are graphs (if applicable) pop up window will open and after review user is expected to close the window for program to complete. Upon completing one program, user will be prompted to run other programs and if user choose to run another program, he/she need to enter Y. Please note the system allows 3 attempts (successful / unsuccessful) to run the program. Upon 3 attempts system will terminate and return to terminal/cmd.
Below is the example of system with various prompts user is expected -    

    > python demo.py
    .
    .
    Please enter key to start the program ->
    .
    .
    We assume all files are under this directory, if NOT please terminate program (CTL+C) else hit enter ->
    .
    .
    [1] for Analytics; [2] for Apriori Algorithm; [3] for Potential Victim : 
    .
    .
    Do you want to run other programs please enter [Y] or [N] :
    >


Features
This project will provide below features -
    1. Provide some analytics on the recent trends on gun violence in Chicago.
    2. It will provide insight on association of one feature on another using Apriori algorithm.
    3. Build model using Decision Tree & SVM to determine potential victim given their age, gender and location.

Status
Project is: Finished

References 
1. https://pandas.pydata.org/
2. http://numpy.scipy.org/
3. https://matplotlib.org/
4. https://seaborn.pydata.org/
5. https://pypi.org/project/apyori/
6. https://scikit-learn.org/stable/
7.

Contact
sbidk002@ucr.edu
