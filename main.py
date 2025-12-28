def show_menu():
    print("\n==============================")
    print(" Deep Learning Model Selector ")
    print("==============================")
    print("1. CNN (Image Classification)")
    print("2. RNN (IMDB Sentiment Analysis)")
    print("3. LSTM (Time Series Prediction)")
    print("4. Bidirectional LSTM (Stock Prediction)")
    print("5. Netflix Stock Price Prediction (Bi-LSTM)")
    print("q. Exit")
    print("==============================\n")


def main():
    while True:
        show_menu()
        choice = input("Enter your choice (0-5): ").strip()

        if choice == "1":
            from Models import CNN
            print("\n[RUNNING] CNN Model\n")
            CNN.run()

        elif choice == "2":
            from Models import RNN
            print("\n[RUNNING] RNN Model\n")
            RNN.run()

        elif choice == "3":
            from Models import LSTM
            print("\n[RUNNING] LSTM Model\n")
            LSTM.run()

        elif choice == "4":
            from Models import Bidirectional_LSTM
            print("\n[RUNNING] Bidirectional LSTM Model\n")
            Bidirectional_LSTM.run()

        elif choice == "5":
            from Models import Netflix_Stock_Price_Prediction
            print("\n[RUNNING] Netflix Stock Price Prediction Model\n")
            Netflix_Stock_Price_Prediction.run()
        
        elif choice == "q".lower():
            print("\nExiting Deep Learning Program. Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select between 0 and 4.\n")


if __name__ == "__main__":
    main()
