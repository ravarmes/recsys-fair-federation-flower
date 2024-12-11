import os

# # FedAvg --------------------------------------------------------------------------------------
os.system("python FedAvg-Example-32-08.py 2> FedAvg-Example-32-08-03.txt")
os.system("python FedAvg-Example-32-08.py 2> FedAvg-Example-32-08-04.txt")
os.system("python FedAvg-Example-32-08.py 2> FedAvg-Example-32-08-05.txt")

# # Fed(l) --------------------------------------------------------------------------------------
os.system("python FedAvg-Loss-32-08.py 2> FedAvg-Loss-32-08-03.txt")
os.system("python FedAvg-Loss-32-08.py 2> FedAvg-Loss-32-08-04.txt")
os.system("python FedAvg-Loss-32-08.py 2> FedAvg-Loss-32-08-05.txt")

# FairFed(l) ACTIVITY -------------------------------------------------------------------------
os.system("python FedFair-Loss-Activity-32-08.py 2> FedFair-Loss-Activity-32-08-03.txt")
os.system("python FedFair-Loss-Activity-32-08.py 2> FedFair-Loss-Activity-32-08-04.txt")
os.system("python FedFair-Loss-Activity-32-08.py 2> FedFair-Loss-Activity-32-08-05.txt")