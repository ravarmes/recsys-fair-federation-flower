import os

os.system("python FedFair-Loss-Activity-Gradiente.py 2> FedFair-Loss-Activity-Gradiente-Taxa040-01.txt")
os.system("python FedFair-Loss-Activity-Gradiente.py 2> FedFair-Loss-Activity-Gradiente-Taxa040-02.txt")

os.system("python FedFair-Loss-Age-Gradiente.py 2> FedFair-Loss-Age-Gradiente-Taxa040-01.txt")
os.system("python FedFair-Loss-Age-Gradiente.py 2> FedFair-Loss-Age-Gradiente-Taxa040-02.txt")

os.system("python FedFair-Loss-Gender-Gradiente.py 2> FedFair-Loss-Gender-Gradiente-Taxa040-01.txt")
os.system("python FedFair-Loss-Gender-Gradiente.py 2> FedFair-Loss-Gender-Gradiente-Taxa040-02.txt")

