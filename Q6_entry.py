from Q6_MaxEntropy.entropy_opt import MaxEntropy

if __name__ == "__main__":
    name_list = ['zachary', 'dolphins', 'jazz']
    budget_list = [40, 40, 1000]
    interval_list = [2, 2, 50]
    for i in range(len(name_list)):
        max_entropy = MaxEntropy(f'datasets/{name_list[i]}.gml', budget=budget_list[i], interval=interval_list[i], method='algebraic connectivity')
        max_entropy.run()