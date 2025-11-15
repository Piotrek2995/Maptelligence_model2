import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj plik CSV
df = pd.read_csv("data/winequality-red.csv", sep=";")

def main():
    # Sprawdź, czy są jakieś wartości puste/NaN/NULL
    if df.isnull().values.any():
        print("Są wartości NULL / NaN / puste")
        print(df.isnull().sum())  # ile brakuje w każdej kolumnie
    else:
        print("Brak wartości NULL")

    print(df.info())
    df.hist(bins=30, figsize=(10, 8))
    plt.show()
    print(df.head())

    bins = 30
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols].hist(bins=bins, figsize=(14, 10), grid=False)
    plt.show()

if __name__ == "__main__":
    main()