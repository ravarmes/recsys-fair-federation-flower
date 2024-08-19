import os

# FedAvg --------------------------------------------------------------------------------------
# os.system("python FedRecSysFair-FedAVG.py 2> FedRecSysFair-FedAVG-1.txt")
# os.system("python FedRecSysFair-FedAVG.py 2> FedRecSysFair-FedAVG-2.txt")
# os.system("python FedRecSysFair-FedAVG.py 2> FedRecSysFair-FedAVG-3.txt")
# os.system("python FedRecSysFair-FedAVG.py 2> FedRecSysFair-FedAVG-4.txt")

# Fed(l) --------------------------------------------------------------------------------------
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-4.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-5.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-6.txt")

# FairFed(l) ACTIVITY -------------------------------------------------------------------------
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity-4.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity-5.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Activity-6.txt")

# FairFed(l) AGE ------------------------------------------------------------------------------
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age-4.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age-5.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Age-6.txt")

# FairFed(l) GENDER ---------------------------------------------------------------------------
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender-4.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender-5.txt")
os.system("python FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender.py 2> FedRecSysFair-FedCustom-Aggregate-Loss-Fair-Gender-6.txt")


