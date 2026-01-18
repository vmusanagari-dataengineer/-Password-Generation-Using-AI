from bin.Classifier import predict_class
from bin.Generator import generate_passwords
import os
import time

def strong_password():
    password = generate_passwords()
    while predict_class(password) < 4:
        password = generate_passwords()
    return password

labels = ['very weak', 'weak', 'normal', 'strong', 'very strong']
repeat = "yes"
while repeat.lower() == "yes":

    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')  # For Windows: 'cls', For others: 'clear'

    # Display menu
    print("=" * 40)
    print("Welcome to the Password Utility Tool")
    print("=" * 40)
    print("\n1. Password Classifier")
    print("2. Strong Password Generator\n")

    # Take user's choice
    choice = input("Enter your choice (1 or 2): ")

    # Perform actions based on choice
    if choice == "1":
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen for cleaner input
        password = input("Enter the password you want to classify:\n>>> ")
        password_class = labels[predict_class(password)]  # Get the classification
        print(f"\nResult:\n'{password}' is classified as a '{password_class}' password.")
    elif choice == "2":
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen for cleaner output
        generated_password = strong_password()
        print("\nStrong Password Generated Successfully!")
        print(">>>", generated_password)
    else:
        print("\nInvalid choice. Please try again!")

    # Ask if the user wants to repeat
    print("\n" + "-" * 40)
    repeat = input("Do you want to perform another operation? (yes to repeat, anything else to exit): ").strip()
    
    # Optional pause before restarting
    time.sleep(1)
