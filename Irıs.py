import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("iris.csv")

df.drop(['Id'], axis=1, inplace=True)


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "KNN": KNeighborsClassifier(),
    "Karar Ağacı": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(kernel="linear"),
    "Lojistik Regresyon":LogisticRegression(),
    "Yapay Sinir Ağı": MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1500)
}

for name, model in models.items():
    if name in ["KNN","SVM","Lojistik Regresyon","Yapay Sinir Ağı"]:
        model.fit(X_train_scaled,y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)


    acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    train_acc = accuracy_score(y_train,y_train_pred)
    test_acc = accuracy_score(y_test,y_test_pred)

    print(f"{name}: Train Acc:{train_acc*100:.2f}%, Test Acc:{test_acc*100:.2f}%")

    print(f"📌 {name} Modeli:")
    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(f"✅ Precision: {precision * 100:.2f}%")
    print(f"✅ Recall: {recall * 100:.2f}%")
    print(f"✅ F1 Score: {f1 * 100:.2f}%")
    print("-" * 50)

    if name in ["Karar Ağacı", "Random Forest"]:
        feature_importances = model.feature_importances_
        feature_names = df.columns[:-1]  # Son sütun hedef değişken olduğu için çıkarıyoruz
        importance_df = pd.DataFrame({"Özellik": feature_names, "Önem Skoru": feature_importances})
        importance_df = importance_df.sort_values(by="Önem Skoru", ascending=False)

        print(f"📊 {name} için Özellik Önem Skorları:")
        print(importance_df)
        print("=" * 50)
