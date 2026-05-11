import tensorflow as tf 
import pandas as pd
import os
import numpy as np

file_path = "gpaScore.xls"

if os.path.exists(file_path):
    try: 
        # 1. 데이터 로드 및 열 이름 전처리
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip().str.lower() # 공백 제거 및 소문자화
        data = data.dropna() # 빈칸 제거
        
        print("✅ 데이터 로드 완료!")
        print("현재 열 이름들:", data.columns.tolist())

        # 2. x_train, y_train 정의 (에러 해결 핵심!)
        # x_train: 입력 데이터 (gre, gpa, rank)
        # y_train: 정답 데이터 (admit)
        x_train = []
        for i, rows in data.iterrows():
            x_train.append([rows['gre'], rows['gpa'], rows['rank']])
        
        y_train = data['admit'].values

        # 3. 모델 설계
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(128, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid") # 합격 확률은 0~1 사이이므로 sigmoid
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

        # 4. 학습 실행
        # 리스트를 넘파이 배열로 바꿔서 넣어줍니다.
        model.fit(np.array(x_train), np.array(y_train), epochs=100)
        
        print("🎉 학습 성공!")

    except Exception as e:
        print("❌ 데이터 처리 실패:", str(e))
else:
    print("❌ 파일이 존재하지 않습니다:", file_path)