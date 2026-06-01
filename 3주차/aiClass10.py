#!pip install deap

import random
import numpy as np
from deap import base, creator, tools, algorithms

# 1. 환경 설정: 데이터 생성 (Target: 3x^2 + 2x + 5)
def target_function(x):
    return 3*x**2 + 2*x + 5

x_train = np.linspace(-10, 10, 100)
y_train = target_function(x_train)

# 2. DEAP 설정
# 적합도(Fitness): 오차를 최소화해야 하므로 weights를 음수로 설정 (Minimize)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 개체(Individual): 3개의 계수(a, b, c)를 담는 리스트
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 유전자 초기화: -10에서 10 사이의 실수를 랜덤하게 생성
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. 평가 함수 (MSE: 평균 제곱 오차 계산)
def evaluate(individual):
    a, b, c = individual
    y_predict = a*x_train**2 + b*x_train + c
    mse = np.mean((y_train - y_predict)**2)
    return (mse,)

toolbox.register("evaluate", evaluate)
# 교차(Crossover): 두 개체의 유전자를 섞음
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# 변이(Mutation): 가우시안 변이 적용
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
# 선택(Selection): 토너먼트 방식
toolbox.register("select", tools.selTournament, tournsize=3)

# 4. 메인 루프
def main():
    random.seed(42)
    
    # 초기 인구 생성 (300명)
    pop = toolbox.population(n=300)
    
    # 명예의 전당 (가장 우수한 개체 보관)
    hof = tools.HallOfFame(1)
    
    # 통계 도구 설정
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # 유전자 알고리즘 실행
    # ngen: 세대 수, cxpb: 교차 확률, mutpb: 변이 확률
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, 
                        stats=stats, halloffame=hof, verbose=True)
    
    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
    
    # 결과 출력
    best_ind = hof[0]
    print(f"\n최적의 계수: a={best_ind[0]:.4f}, b={best_ind[1]:.4f}, c={best_ind[2]:.4f}")
    print(f"최종 오차(MSE): {best_ind.fitness.values[0]:.6f}")
