import os

# FedAvg --------------------------------------------------------------------------------------
# os.system("python FedAvg-Example-16-8.py 2> FedAvg-Example-16-8-03.txt")
os.system("python FedAvg-Example-16-8.py 2> FedAvg-Example-16-8-04.txt")

# Fed(l) --------------------------------------------------------------------------------------
# os.system("python FedAvg-Loss-16-8.py 2> FedAvg-Loss-16-8-03.txt")
# os.system("python FedAvg-Loss-16-8.py 2> FedAvg-Loss-16-8-04.txt")

# # FairFed(l) ACTIVITY -------------------------------------------------------------------------
os.system("python FedFair-Loss-Activity-16-8.py 2> FedFair-Loss-Activity-16-8-03.txt")
os.system("python FedFair-Loss-Activity-16-8.py 2> FedFair-Loss-Activity-16-8-04.txt")

# # FairFed(l) AGE ------------------------------------------------------------------------------
os.system("python FedFair-Loss-Age-16-8.py 2> FedFair-Loss-Age-16-8-03.txt")
os.system("python FedFair-Loss-Age-16-8.py 2> FedFair-Loss-Age-16-8-04.txt")

# # FairFed(l) GENDER ---------------------------------------------------------------------------
os.system("python FedFair-Loss-Gender-16-8.py 2> FedFair-Loss-Gender-16-8.py-03.txt")
os.system("python FedFair-Loss-Gender-16-8.py 2> FedFair-Loss-Gender-16-8.py-04.txt")
