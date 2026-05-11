import numpy as np
import pandas as pd
import os
import tensorflow as tf

file_path = "gpaScore.xls"
# pandas로 엑셀 파일들을 불러올떄 어떻게 불러오는지임 
if os.path.exists(file_path):
    try: 
        # 공백을 제거 엑셀 데이터를 불러올때는 공백을 제거하고 불러와야함.. 귀찮지만 그래야해..
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip().str.lower()
        data = data.dropna()
        print("✅ 데이터 로드 완료!")
        print("-" * 30)
        print(data)
        y_data = data['admit'].values   
        print("admin 데이터:",y_data)
        print("✅ y_data 추출 완료!")
        
        print("-" * 30)
        data.columns = data.columns
        X_train = []
        y_train = data['admit'].values
        print("Y_train 데이터:", y_train)

        for i, rows in data.iterrows():
            X_train.append([rows['gre'], rows['gpa'], rows['rank']])
            print("Xtrain 데이터", X_train)

        # 반복문으로 데이터를 반복하고, iterrows는 데이터의 각행을 가져오는 것입니다. 
        # for i, rows in data.iterrows():
        #     print(f"행 {i} 데이터", rows['gre'])
        #     X_train.append([rows['gre'], rows['gpa'], rows['rank']])
        # exit() 학습 연구과정입니다. 어떻게 학습하는지에 따라서 달라지는 것입니다. 
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(128, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid") # 합격 확률은 0~1 사이이므로 sigmoid
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        # gpa 열 데이터 뽑아 보기 
        
        # 데이터를 학습시키는 용도임 
        # np.array로 리스트를 넘파이 배열로 바꿔서 넣어줍니다. 리스트 처럼 조작을 할 수 있음
        model.fit(np.array(X_train), np.array(y_train), epochs=100)
        print("🎉 학습 성공!")

        # loss, accuracy 는 손실 값이 적으면 적을 수록 값이 좋아지는 것이고, 
        # accuracy는 정확도를 나타내는 것이며, 1에 가까울 수록 정확도가 높아지는 것입니다. 

        # 자 이제 예측 

        test_data = np.array([[320, 3.5, 2], [755, 3.1, 11]]) # 예시 입력 데이터 (gre=320, gpa=3.5, rank=2)

        predit = model.predict(test_data)
        print("예측 결과 (합격 확률):", predit)

    except Exception as e:
        print(f"❌ 읽기 실패: {e}")
else:
    print("❌ 파일을 찾을 수 없습니다. 경로가 'project1' 폴더 안이 맞는지 확인해 보세요!")
